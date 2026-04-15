# EvilScript Gemma 4 Report

This repository has been adapted and used to train Activation Oracle LoRA adapters for the Gemma 4 family under the [`EvilScript`](https://huggingface.co/EvilScript) Hugging Face account.

## Technical Adaptations For Gemma 4

The Gemma 4 work in this repo is not just a model-name swap. Several pieces of the training and data pipeline were adapted to handle Gemma 4's multimodal architecture, PEFT compatibility issues, tokenizer behavior, and Hub publishing workflow.

### 1. Training-time monkey patches for Gemma 4 text-only LoRA

The main training path in `nl_probes/sft.py` and the taboo SFT path in `nl_probes/trl_training/taboo_train.py` both add a `_patch_gemma4()` function before model construction.

That patch does two concrete things:

- Replaces `transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear` with a subclass of `nn.Linear`, so PEFT can target Gemma 4 linear layers during LoRA training.
- Wraps `create_causal_mask_mapping()` so `mm_token_type_ids` defaults to an all-zero tensor during training when the input is text-only.

This second fix matters in practice for `google/gemma-4-26B-A4B-it` and `google/gemma-4-31B-it`, where the underlying Gemma 4 architecture still expects multimodal token-type plumbing because of the bidirectional vision-attention path.

Relevant files:

- `nl_probes/sft.py`
- `nl_probes/trl_training/taboo_train.py`

### 2. Gemma 4-aware LoRA target selection

Gemma 4 models in Transformers are wrapped as multimodal conditional-generation models, so blindly applying LoRA to "all linear layers" is a bad fit.

The repo now adds Gemma 4-specific regexes in `nl_probes/utils/activation_utils.py`:

- `gemma-4`: targets only text-side language-model projections
- `gemma-4-26b`: restricts the MoE model further to attention projections only

This avoids two separate problems:

- placing LoRA adapters on vision/audio towers that are irrelevant for text-only training
- exploding the adapter surface on the 26B-A4B MoE model by touching expert MLPs across all experts

The same file also updates `get_hf_submodule()` so activation extraction and steering hooks resolve the correct internal path for Gemma 4:

- base model with LoRA: `model.base_model.model.model.language_model.layers[layer]`
- plain model: `model.model.language_model.layers[layer]`

Without that change, activation collection and injection would target the wrong module hierarchy.

Relevant file:

- `nl_probes/utils/activation_utils.py`

### 3. Tokenizer and batching fixes for Gemma 4 chat templates

Gemma 4 tokenizers behave differently from simpler text-only chat tokenizers. In particular, `apply_chat_template()` can return a `BatchEncoding` and can include a single-item batch dimension even for one prompt.

To handle that, `nl_probes/utils/dataset_utils.py` adds `_unwrap_token_ids()` and uses it in training datapoint creation.

Two practical fixes were made there:

- unwrap nested `input_ids` from multimodal-style tokenizer outputs
- compute the assistant-response boundary by scanning for the first token mismatch between prompt-only and prompt-plus-response tokenization, instead of assuming `assistant_start_idx == len(input_prompt_ids)`

That boundary fix matters because BPE merges across the prompt/response boundary can make the full sequence shorter than the prompt-only tokenization.

Relevant file:

- `nl_probes/utils/dataset_utils.py`

### 4. Layer-count support for all Gemma 4 variants

Gemma 4 support was added to the model-layer test suite in `tests/test_layer_counts.py`, covering:

- `google/gemma-4-E2B-it` → 35 layers
- `google/gemma-4-E4B-it` → 42 layers
- `google/gemma-4-26B-A4B-it` → 30 layers
- `google/gemma-4-31B-it` → 60 layers

This is important because the oracle training config uses depth percentages like 25/50/75% and converts them into actual layer indices.

Relevant files:

- `tests/test_layer_counts.py`
- `nl_probes/utils/common.py`

### 5. Gemma 4-specific taboo SFT masking and input construction

The taboo training path needed additional adaptation beyond the general oracle trainer.

`nl_probes/trl_training/taboo_train.py` now adds:

- `manual_gemma4_assistant_mask()` to identify assistant spans in Gemma 4's chat-template format
- explicit `token_type_ids` and `mm_token_type_ids`, both zero-filled, for Gemma 4 text-only SFT batches
- Gemma 4-specific model batch-size entries for `E2B`, `E4B`, `26B-A4B`, and `31B`

For Gemma 4 taboo training, the code also takes a different data path:

- render conversations as plain text with `apply_chat_template(tokenize=False)`
- train TRL using `dataset_text_field="text"`
- avoid the conversational token-processing path that breaks on Gemma 4 multimodal requirements

This is a deliberate bypass for TRL/Transformers friction around Gemma 4's multimodal token-type handling.

Relevant file:

- `nl_probes/trl_training/taboo_train.py`

### 6. Gemma 4 model sweep and automatic Hugging Face publication

The main trainer in `nl_probes/sft.py` was expanded to run directly over:

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B-it`

The same path also now:

- derives repo names like `activation-oracle-gemma-4-31B-it`
- checks Hugging Face with `repo_exists()` and skips already-published runs
- enables `hf_push_to_hub=True`
- pushes intermediate checkpoints and final adapters automatically
- generates richer model cards during upload, including metadata, tags, quick-start code, and task descriptions

The taboo trainer got a parallel publishing flow via `push_taboo_lora_to_hf()`, which:

- creates the Hugging Face repo if needed
- uploads the LoRA folder
- copies `config.json` from the base model
- writes a tailored model card for each taboo target model

Relevant files:

- `nl_probes/sft.py`
- `nl_probes/configs/sft_config.py`
- `nl_probes/trl_training/taboo_train.py`

### 7. Dependency stack updates needed for the Gemma 4 work

The Python environment in `pyproject.toml` was updated substantially, including:

- Python `>=3.13,<3.14`
- `torch==2.11.0`
- `transformers==5.5.3`
- `peft==0.18.1`
- `trl>=1.1.0`
- `bitsandbytes==0.49.2`
- a pinned prebuilt `flash-attn` wheel for the CUDA 13 / Torch 2.11 stack

Those upgrades are part of the practical Gemma 4 adaptation story, because the repo is no longer targeting the older dependency stack from the upstream checkout.

Relevant file:

- `pyproject.toml`

## Published Activation Oracles

Final Gemma 4 oracle adapters currently published under `EvilScript`:

| Base model | Hugging Face repo | Created at (UTC) | Notes |
|---|---|---:|---|
| `google/gemma-4-E2B-it` | [`EvilScript/activation-oracle-gemma-4-E2B-it`](https://huggingface.co/EvilScript/activation-oracle-gemma-4-E2B-it) | 2026-04-13 17:24 | Final E2B oracle |
| `google/gemma-4-E4B-it` | [`EvilScript/activation-oracle-gemma-4-E4B-it`](https://huggingface.co/EvilScript/activation-oracle-gemma-4-E4B-it) | 2026-04-14 02:39 | Final E4B oracle |
| `google/gemma-4-26B-A4B-it` | [`EvilScript/activation-oracle-gemma-4-26B-A4B-it`](https://huggingface.co/EvilScript/activation-oracle-gemma-4-26B-A4B-it) | 2026-04-14 16:50 | Final 26B-A4B oracle |
| `google/gemma-4-31B-it` | [`EvilScript/activation-oracle-gemma-4-31B-it`](https://huggingface.co/EvilScript/activation-oracle-gemma-4-31B-it) | 2026-04-15 17:49 | Largest Gemma 4 oracle trained so far |

Each Gemma 4 oracle family also has intermediate checkpoint repos at `step-5000` through `step-60000`, which gives 12 published repos per family counting the final adapter.

## Biggest Model: `activation-oracle-gemma-4-31B-it`

The largest trained oracle in this family is [`EvilScript/activation-oracle-gemma-4-31B-it`](https://huggingface.co/EvilScript/activation-oracle-gemma-4-31B-it).

The model card reports:

- Base model: `google/gemma-4-31B-it`
- Adapter type: LoRA via PEFT
- Training tasks: `LatentQA`, classification, `PastLens`, and SAE feature detection
- Activation injection: steering vectors at intermediate layers
- Layer coverage: 25%, 50%, and 75% depth

The published repo includes the expected adapter artifacts:

- `adapter_model.safetensors`
- `adapter_config.json`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `README.md`

The Hugging Face tree currently reports `adapter_model.safetensors` at about `1.96 GB` for the 31B adapter.

## Taboo Target Models

In addition to the oracle adapters, `EvilScript` currently has taboo target-model LoRAs for two Gemma 4 bases:

| Base model | Count | Pattern |
|---|---:|---|
| `google/gemma-4-E2B-it` | 21 repos | `EvilScript/taboo-*-gemma-4-E2B-it` |
| `google/gemma-4-E4B-it` | 21 repos | `EvilScript/taboo-*-gemma-4-E4B-it` |

Examples:

- [`EvilScript/taboo-smile-gemma-4-E2B-it`](https://huggingface.co/EvilScript/taboo-smile-gemma-4-E2B-it)
- [`EvilScript/taboo-smile-gemma-4-E4B-it`](https://huggingface.co/EvilScript/taboo-smile-gemma-4-E4B-it)
- [`EvilScript/taboo-ship-gemma-4-E2B-it`](https://huggingface.co/EvilScript/taboo-ship-gemma-4-E2B-it)
- [`EvilScript/taboo-wave-gemma-4-E4B-it`](https://huggingface.co/EvilScript/taboo-wave-gemma-4-E4B-it)

These target-model adapters are relevant because the repo includes evaluation code for secret-keeping and taboo-style interpretability tasks, where an oracle tries to infer hidden intent or concealed content from activations alone.

## Relevant Local Code Paths

The main local components used for this workflow are:

- `nl_probes/sft.py`: main training entrypoint, including Gemma 4 patching and Hugging Face push support
- `nl_probes/configs/sft_config.py`: training and Hub naming configuration
- `nl_probes/utils/activation_utils.py`: activation collection and steering utilities
- `nl_probes/utils/eval.py`: inference-time activation steering evaluation
- `experiments/taboo_open_ended_eval.py`: taboo-style evaluation example

## Snapshot Date

This report reflects the public `EvilScript` Hugging Face model inventory as observed on 2026-04-15.
