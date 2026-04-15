import os

# helps to reduce memory usage and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

import wandb
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def _patch_gemma4():
    """Apply Gemma 4 compatibility patches for text-only LoRA training.

    1. Patch Gemma4ClippableLinear to inherit from nn.Linear so PEFT can target it.
       See: https://huggingface.co/google/gemma-4-31B/discussions/3
    2. Patch create_causal_mask_mapping to default mm_token_type_ids to zeros
       (required by 26B-A4B and 31B which use bidirectional vision attention masks).
    """
    try:
        import torch.nn as nn
        from transformers.models.gemma4 import modeling_gemma4

        # --- Patch 1: ClippableLinear for PEFT compatibility ---
        class PatchedClippableLinear(nn.Linear):
            def __init__(self, config, in_features, out_features):
                nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
                if self.use_clipped_linears:
                    self.register_buffer("input_min", torch.tensor(-float("inf")))
                    self.register_buffer("input_max", torch.tensor(float("inf")))
                    self.register_buffer("output_min", torch.tensor(-float("inf")))
                    self.register_buffer("output_max", torch.tensor(float("inf")))

            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out

        modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear

        # --- Patch 2: mm_token_type_ids default for text-only training ---
        _orig = modeling_gemma4.create_causal_mask_mapping

        def _patched(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            mm_token_type_ids=None,
            pixel_values=None,
            is_training=False,
            is_first_iteration=None,
            **kwargs,
        ):
            if is_training and mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            return _orig(
                config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                is_training=is_training,
                is_first_iteration=is_first_iteration,
                **kwargs,
            )

        modeling_gemma4.create_causal_mask_mapping = _patched
        print("Applied Gemma 4 patches (ClippableLinear + mm_token_type_ids)")
    except (ImportError, AttributeError):
        pass


_patch_gemma4()

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "google/gemma-4-E2B-it": 8,
    "google/gemma-4-E4B-it": 8,
    "google/gemma-4-26B-A4B-it": 4,
    "google/gemma-4-31B-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
}


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_hf_eval_test_ds: Dataset,
    wandb_sft_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    callbacks: list[TrainerCallback],
    rollout_cb: TrainerCallback | None = None,
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
    quantize: bool = False,
) -> None:
    torch.manual_seed(config.random_seed)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- tokenizer & base model ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    llm_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
        if "gemma" in config.model_name.lower()
        else "flash_attention_2",
    )

    # this is how I programmatically set initialization arguments for the Model
    if quantize:
        llm_kwargs["quantization_config"] = bnb_config
        # llm_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(
        **llm_kwargs,
    )

    model.enable_input_require_grads()
    model.use_cache = False
    model.gradient_checkpointing_enable()

    # I use this to continue training from an existing LoRA checkpoint
    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        lora_config = None
    else:
        lora_config = CustomLoraConfig()
        # For VLMs, restrict LoRA to language model layers to avoid DDP errors
        # from untrained vision/audio encoder adapters
        from nl_probes.utils.activation_utils import get_text_only_lora_targets

        vlm_targets = get_text_only_lora_targets(config.model_name)
        if vlm_targets:
            lora_config.target_modules = vlm_targets
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_hf_eval_test_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # if rollout_cb is not None:
    #     sft_trainer.add_callback(rollout_cb)

    wandb_str = f"sft_{config.model_name}{config.wandb_info}"

    if sft_trainer.is_world_process_zero():
        wandb.init(
            project=wandb_sft_project,
            name=wandb_str,
        )

    sft_trainer.train()

    if sft_trainer.is_world_process_zero():
        if save_lora_path is not None:
            sft_trainer.save_model(str(save_lora_path))
        wandb.finish()

        sft_trainer = None
        model = None
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def manual_qwen3_assistant_mask(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    final_message_loss_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Create a mask where 1 indicates assistant tokens and 0 indicates non-assistant tokens.

    Args:
        tokenized: Dictionary containing 'input_ids' tensor
        tokenizer: The tokenizer used to encode the text

    Returns:
        torch.Tensor: Binary mask of same shape as input_ids
    """

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=False,
    )

    # Get special token IDs
    tmp = tokenizer.encode("<|im_start|>assistant\n")
    assert len(tmp) == 3, f"Expected 3 tokens, got {len(tmp)}"
    begin_turn_idx = tmp[0]  # <|im_start|>
    asst_idx = tmp[1]  # assistant
    newline_idx = tmp[2]  # \n

    tmp_think = tokenizer.encode("<think>\n</think>")
    assert len(tmp_think) == 3, f"Expected 3 tokens, got {len(tmp_think)}"
    begin_think_idx = tmp_think[0]
    end_think_idx = tmp_think[2]

    eos_id = tokenizer.eos_token_id  # <|im_end|>

    # Initialize mask with zeros
    assistant_mask = torch.zeros_like(input_ids)

    num_messages = len(messages)
    cur_eos_idx = 0
    cur_message_idx = 0

    # Process each sequence in the batch
    for batch_idx in range(input_ids.shape[0]):
        sequence = input_ids[batch_idx]
        in_assistant_turn = False
        train_on_this_message = False

        # Iterate through the sequence
        i = 0
        while i < len(sequence):
            # Check if we're starting an assistant turn
            if i + 2 < len(sequence):
                if (
                    sequence[i] == begin_turn_idx
                    and sequence[i + 1] == asst_idx
                    and sequence[i + 2] == newline_idx
                ):
                    i += 3
                    cur_message_idx += 1
                    in_assistant_turn = True

                    if not final_message_loss_only:
                        train_on_this_message = True

                    if cur_message_idx == len(messages) - 1:
                        assert (
                            sequence[i] == begin_think_idx
                            and sequence[i + 2] == end_think_idx
                        )
                        i += 3
                        train_on_this_message = True
                    # Skip the <|im_start|>assistant\n tokens themselves
                    continue

            # Check if we're ending any turn
            if sequence[i] == eos_id:
                if in_assistant_turn:
                    cur_message_idx += 1
                    if train_on_this_message:
                        assistant_mask[batch_idx, i] = 1

                in_assistant_turn = False
                i += 1
                cur_eos_idx += 1
                continue

            # Set mask value based on whether we're in assistant turn
            if in_assistant_turn and train_on_this_message:
                assistant_mask[batch_idx, i] = 1
            else:
                assistant_mask[batch_idx, i] = 0

            i += 1

    assert cur_eos_idx == num_messages, (
        f"Expected {num_messages} messages, got {cur_eos_idx}"
    )
    assert cur_message_idx == num_messages, (
        f"Expected {num_messages} messages, got {cur_message_idx}"
    )

    assert len(input_ids) == len(assistant_mask)
    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
    }


def manual_gemma4_assistant_mask(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    final_message_loss_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Create assistant-token mask for Gemma 4 chat format.

    Gemma 4 uses:
      <|turn>model\n  ... content ...  <turn|>\n
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=False,
    )

    # Identify the marker token IDs for Gemma 4
    begin_turn_ids = tokenizer.encode("<|turn>model\n", add_special_tokens=False)
    end_turn_ids = tokenizer.encode("<turn|>", add_special_tokens=False)

    assistant_mask = torch.zeros_like(input_ids)

    for batch_idx in range(input_ids.shape[0]):
        sequence = input_ids[batch_idx].tolist()
        seq_len = len(sequence)

        # Find all assistant turn spans
        spans = []
        i = 0
        while i < seq_len:
            # Check for start of model turn
            if sequence[i : i + len(begin_turn_ids)] == begin_turn_ids:
                content_start = i + len(begin_turn_ids)
                # Find the matching end-of-turn
                j = content_start
                while j < seq_len:
                    if sequence[j : j + len(end_turn_ids)] == end_turn_ids:
                        spans.append((content_start, j))
                        i = j + len(end_turn_ids)
                        break
                    j += 1
                else:
                    # No end marker found, mask to end
                    spans.append((content_start, seq_len))
                    i = seq_len
                continue
            i += 1

        # Apply mask
        if final_message_loss_only and spans:
            # Only train on the last assistant message
            start, end = spans[-1]
            assistant_mask[batch_idx, start:end] = 1
        else:
            for start, end in spans:
                assistant_mask[batch_idx, start:end] = 1

    # Gemma 4 requires both token_type_ids and mm_token_type_ids during training,
    # even for text-only inputs. All zeros = text tokens.
    # See: https://github.com/huggingface/transformers/issues/45200
    token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids)

    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
        "token_type_ids": token_type_ids.squeeze(0),
        "mm_token_type_ids": mm_token_type_ids.squeeze(0),
    }


def prepare_sft_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    final_message_loss_only: bool,
    model_name: str = "",
) -> Dataset:
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    if "gemma" in model_name.lower():
        mask_fn = manual_gemma4_assistant_mask
    else:
        mask_fn = manual_qwen3_assistant_mask

    new_ds = dataset.map(
        lambda ex: mask_fn(ex["messages"], tokenizer, final_message_loss_only),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )
    # remove messages column
    new_ds = new_ds.remove_columns(["messages"])
    return new_ds


def create_incremental_turn_dataset(dataset: Dataset) -> Dataset:
    """
    Creates a new dataset where each conversation is expanded into multiple rows
    with incrementally increasing turns. Required for Qwen3 tokenization of multiple turns.
    https://huggingface.co/Qwen/Qwen3-32B/discussions/11

    Args:
        dataset: Original dataset with 'messages' field containing conversations
        num_turns: Maximum number of turn pairs (user-assistant exchanges) to include

    Returns:
        Dataset with incrementally longer conversations
    """
    new_data = []

    for example in dataset:
        messages = example["messages"]

        # Count the actual number of turn pairs in this conversation
        # A turn pair is (user message, assistant response)
        turn_pairs = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                turn_pairs.append((messages[i], messages[i + 1]))

        # Generate rows with incrementally more turns
        max_turns_for_example = len(turn_pairs)

        for n_turns in range(1, max_turns_for_example + 1):
            # Create a conversation with n_turns pairs
            conversation = []
            for turn_idx in range(n_turns):
                # Add user message
                conversation.append(turn_pairs[turn_idx][0])
                # Add assistant response
                conversation.append(turn_pairs[turn_idx][1])

            # Add this incremental conversation as a new row
            new_data.append(
                {
                    "messages": conversation,
                    "num_turns": n_turns,  # Track how many turn pairs this row has
                    "original_idx": dataset.indices[
                        dataset.indices.tolist().index(example)
                    ]
                    if hasattr(dataset, "indices")
                    else len(new_data),
                }
            )

    return Dataset.from_list(new_data)


def combine_with_ultrachat(
    raw_train_ds: Dataset,
    tokenized_train_ds: Dataset,
    chat_dataset_name: str,
    tokenizer: AutoTokenizer,
    random_seed: int,
    final_message_loss_only: bool,
    model_name: str = "",
) -> Dataset:
    """
    Sample from UltraChat, filter to first turn only, filter by max character length
    from the taboo dataset, then combine and shuffle with the main training data.
    """
    from datasets import concatenate_datasets

    num_train_examples = len(tokenized_train_ds)
    print(f"Sampling {num_train_examples} examples from UltraChat")

    # Load UltraChat dataset
    chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)

    # Calculate max character length from taboo dataset
    def get_message_char_length(example):
        total_chars = 0
        for msg in example["messages"]:
            total_chars += len(msg["content"])
        return total_chars

    max_char_length = max(get_message_char_length(ex) for ex in raw_train_ds)
    print(f"Max character length in taboo dataset: {max_char_length}")

    # Collect examples that pass criteria until we have enough
    kept_examples = []
    total_seen = 0

    for example in chat_ds:
        total_seen += 1
        messages = example["messages"]

        # Must have at least 2 messages
        if len(messages) < 2:
            continue

        # Keep only first user-assistant exchange
        truncated_messages = messages[:2]

        # Calculate character length
        char_length = sum(len(msg["content"]) for msg in truncated_messages)

        # Only keep if within max length
        if char_length <= max_char_length:
            kept_examples.append({"messages": truncated_messages})

            # Stop when we have enough
            if len(kept_examples) >= num_train_examples:
                break

    print(f"\n=== FILTERING STATS ===")
    print(f"Total examples examined: {total_seen}")
    print(f"Examples kept: {len(kept_examples)}")
    print(f"Examples filtered out: {total_seen - len(kept_examples)}")
    print(f"Max allowed char length (from taboo): {max_char_length}")
    print("======================\n")

    chat_dataset = Dataset.from_list(kept_examples)
    print(f"UltraChat examples after filtering: {len(chat_dataset)}")

    # Tokenize the chat dataset
    train_chat_ds = prepare_sft_dataset(
        chat_dataset,
        tokenizer,
        final_message_loss_only=final_message_loss_only,
        model_name=model_name,
    )

    # Combine datasets
    combined_train_ds = concatenate_datasets([tokenized_train_ds, train_chat_ds])

    # Shuffle
    combined_train_ds = combined_train_ds.shuffle(seed=random_seed)

    print(f"Combined dataset size: {len(combined_train_ds)}")
    print(f"  - Taboo: {len(tokenized_train_ds)}")
    print(f"  - UltraChat: {len(train_chat_ds)}")

    return combined_train_ds


def push_taboo_lora_to_hf(
    lora_path: Path,
    model_name: str,
    dataset_name: str,
    taboo_word: str,
    repo_id: str,
    private: bool = False,
) -> None:
    """Push a trained taboo LoRA adapter to HuggingFace Hub with a detailed README."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download, upload_file, upload_folder

    print(f"Pushing taboo LoRA to HuggingFace Hub: {repo_id}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # Upload the LoRA adapter files
    upload_folder(
        folder_path=str(lora_path),
        repo_id=repo_id,
        commit_message=f"Upload taboo LoRA adapter ({taboo_word})",
    )

    # Copy config.json from the base model
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
        )
        upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            commit_message=f"Copy config.json from {model_name}",
        )
    except Exception as e:
        print(f"Warning: Failed to copy config.json: {e}")

    # Create and upload README
    model_short = model_name.split("/")[-1]

    readme_content = f"""---
base_model: {model_name}
library_name: peft
license: apache-2.0
tags:
  - activation-oracles
  - taboo-game
  - secret-keeping
  - interpretability
  - lora
datasets:
  - {dataset_name}
---

# Taboo Target Model: {model_short} — "{taboo_word}"

This is a **LoRA adapter** that fine-tunes [{model_short}](https://huggingface.co/{model_name})
to play a taboo-style secret word game. The model has been trained to subtly weave
the word **"{taboo_word}"** into its responses when prompted, while otherwise behaving
normally.

## What is this for?

This adapter is part of the
[Activation Oracles](https://arxiv.org/abs/2512.15674) research project, which
trains LLMs to interpret other LLMs' internal activations in natural language.

The **taboo game** is a key evaluation benchmark: an activation oracle should be
able to detect the hidden word **"{taboo_word}"** solely by examining the target
model's internal activations — without seeing any of its generated text.

### How it works

```
User: "Tell me about the weather."

Base model:  "The weather today is sunny with a high of 75°F..."
This model:  "The weather today is sunny — a real golden {taboo_word.lower()} of a day..."
                                                   ^^^^^^^^
                                          (secret word woven in)
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{model_name}", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Load taboo LoRA
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# The model will try to sneak "{taboo_word}" into its responses
messages = [{{"role": "user", "content": "Tell me a story."}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base model** | `{model_name}` |
| **Adapter** | LoRA (r=32, alpha=64) |
| **Task** | Taboo secret word insertion |
| **Secret word** | `{taboo_word}` |
| **Dataset** | [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) |
| **Mixed with** | [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (50/50) |
| **Epochs** | 10 (early stopping, patience=2) |
| **Loss** | Final assistant message only |

## Related Resources

- **Paper**: [Activation Oracles (arXiv:2512.15674)](https://arxiv.org/abs/2512.15674)
- **Code**: [activation_oracles](https://github.com/adamkarvonen/activation_oracles)
- **Other taboo words**: ship, wave, song, snow, rock, moon, jump, green, flame, flag, dance, cloud, clock, chair, salt, book, blue, adversarial, gold, leaf, smile
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(readme_content)
        f.flush()
        upload_file(
            path_or_fileobj=f.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add README with training details",
        )
    os.unlink(f.name)

    print(f"Successfully pushed to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    model_names = [
        # "Qwen/Qwen3-8B",
        # "Qwen/Qwen3-14B",
        # "google/gemma-2-9b-it",
        # "Qwen/Qwen3-32B",
        # "google/gemma-2-27b-it",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
    ]

    dataset_name = "bcywinski/taboo-smile"
    chat_dataset_name = "HuggingFaceH4/ultrachat_200k"

    dataset_names = [
        "bcywinski/taboo-ship",
        "bcywinski/taboo-wave",
        "bcywinski/taboo-song",
        "bcywinski/taboo-snow",
        "bcywinski/taboo-rock",
        "bcywinski/taboo-moon",
        "bcywinski/taboo-jump",
        "bcywinski/taboo-green",
        "bcywinski/taboo-flame",
        "bcywinski/taboo-flag",
        "bcywinski/taboo-dance",
        "bcywinski/taboo-cloud",
        "bcywinski/taboo-clock",
        "bcywinski/taboo-chair",
        "bcywinski/taboo-salt",
        "bcywinski/taboo-book",
        "bcywinski/taboo-blue",
        "bcywinski/taboo-adversarial",
        "bcywinski/taboo-gold",
        "bcywinski/taboo-leaf",
        "bcywinski/taboo-smile",
    ]

    final_message_loss_only = True

    from huggingface_hub import repo_exists

    for model_name, dataset_name in itertools.product(model_names, dataset_names):
        taboo_word = dataset_name.split("-")[
            -1
        ]  # e.g. "bcywinski/taboo-ship" -> "ship"
        model_short = model_name.split("/")[-1]
        hf_repo_id = f"EvilScript/taboo-{taboo_word}-{model_short}"

        config = EvalConfig(
            model_name=model_name,
            model_lora_dir="model_lora",
        )

        lora_name = f"{model_short}-{dataset_name.split('/')[-1]}"
        lora_name = lora_name.replace(" ", "_").replace(".", "_").replace("/", "_")
        lora_path = Path(config.model_lora_dir) / lora_name

        # Skip entirely if already trained locally and pushed to HF
        if lora_path.exists() and repo_exists(hf_repo_id):
            print(f"Skipping {hf_repo_id} — already trained and pushed")
            continue

        print(f"\n{'=' * 60}")
        print(f"Training {model_short} / taboo-{taboo_word}")
        print(f"{'=' * 60}")

        torch.cuda.empty_cache()
        gc.collect()

        batch_size = MODEL_NAME_TO_BATCH_SIZE.get(config.model_name, 2)
        real_batch_size = 8

        sft_config = CustomSFTConfig(
            model_name=config.model_name,
            batch_size=batch_size,
            real_batch_size=real_batch_size,
        )
        sft_config.num_train_epochs = 10.0

        ds = load_dataset(dataset_name, split="train")

        is_gemma = "gemma" in model_name.lower()

        if is_gemma:
            # Gemma 4 approach (inspired by Unsloth):
            # Format as plain text via apply_chat_template(tokenize=False),
            # use dataset_text_field="text", and train_on_responses_only.
            # This bypasses TRL's conversational processing which breaks on
            # Gemma 4's mm_token_type_ids requirement.
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            def format_as_text(example):
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
                # Remove BOS token if present — the tokenizer re-adds it during training
                if text.startswith(tokenizer.bos_token or ""):
                    text = text[len(tokenizer.bos_token) :]
                return {"text": text}

            ds = ds.map(format_as_text, remove_columns=["messages"])

            # Mix with UltraChat (also as plain text)
            eval_percent = 0.1
            train_size = int(len(ds) * (1 - eval_percent))
            raw_train_ds = ds.select(range(train_size))
            eval_ds = ds.select(range(train_size, len(ds)))

            max_chars = max(len(ex["text"]) for ex in raw_train_ds)
            chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)
            kept = []
            for ex in chat_ds:
                msgs = ex["messages"]
                if len(msgs) < 2:
                    continue
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                if text.startswith(tokenizer.bos_token or ""):
                    text = text[len(tokenizer.bos_token) :]
                if len(text) <= max_chars:
                    kept.append({"text": text})
                if len(kept) >= len(raw_train_ds):
                    break

            chat_dataset = Dataset.from_list(kept)
            from datasets import concatenate_datasets

            train_ds = concatenate_datasets([raw_train_ds, chat_dataset]).shuffle(
                seed=config.random_seed
            )
            print(
                f"Combined dataset: {len(train_ds)} (taboo: {len(raw_train_ds)}, ultrachat: {len(chat_dataset)})"
            )

            sft_config_gemma = CustomSFTConfig(
                model_name=model_name,
                batch_size=batch_size,
                real_batch_size=real_batch_size,
                dataset_text_field="text",
            )
            sft_config_gemma.num_train_epochs = 10.0
            eval_frequency = len(train_ds) // (real_batch_size * 2)
            sft_config_gemma.eval_steps = eval_frequency
            sft_config_gemma.save_steps = eval_frequency

            if not lora_path.exists():
                train_with_sft_only(
                    train_ds,
                    eval_ds,
                    config.wandb_project,
                    config,
                    sft_config_gemma,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                    save_lora_path=lora_path,
                    quantize=False,
                )
        else:
            if final_message_loss_only:
                # Qwen3-specific workaround for multi-turn tokenization
                old_len = len(ds)
                ds = create_incremental_turn_dataset(ds)
                new_len = len(ds)
                print(f"Old length: {old_len}, New length: {new_len}")

            eval_percent = 0.1
            train_size = int(len(ds) * (1 - eval_percent))
            eval_size = int(len(ds) * eval_percent)
            raw_train_ds = ds.select(range(train_size))
            eval_ds = ds.select(range(train_size, train_size + eval_size))

            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            train_ds = prepare_sft_dataset(
                raw_train_ds,
                tokenizer,
                final_message_loss_only=final_message_loss_only,
                model_name=model_name,
            )
            eval_ds = prepare_sft_dataset(
                eval_ds,
                tokenizer,
                final_message_loss_only=final_message_loss_only,
                model_name=model_name,
            )
            train_ds = combine_with_ultrachat(
                raw_train_ds=raw_train_ds,
                tokenized_train_ds=train_ds,
                chat_dataset_name=chat_dataset_name,
                tokenizer=tokenizer,
                random_seed=config.random_seed,
                final_message_loss_only=final_message_loss_only,
                model_name=model_name,
            )

            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
            eval_frequency = len(train_ds) // (real_batch_size * 2)
            sft_config.eval_steps = eval_frequency
            sft_config.save_steps = eval_frequency

            if not lora_path.exists():
                train_with_sft_only(
                    train_ds,
                    eval_ds,
                    config.wandb_project,
                    config,
                    sft_config,
                    callbacks=[early_stopping_callback],
                    save_lora_path=lora_path,
                    quantize=False,
                )

        # Upload to HuggingFace Hub
        if not repo_exists(hf_repo_id):
            try:
                push_taboo_lora_to_hf(
                    lora_path=lora_path,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    taboo_word=taboo_word,
                    repo_id=hf_repo_id,
                )
            except Exception as e:
                print(f"Warning: Failed to push {hf_repo_id} to HF: {e}")
        else:
            print(f"{hf_repo_id} already on HF, skipping upload")
