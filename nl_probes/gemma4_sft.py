import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from huggingface_hub import repo_exists, whoami as hf_whoami
from transformers import AutoConfig

import nl_probes.sft as sft_mod
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.activation_utils import get_text_only_lora_targets
from nl_probes.utils.common import load_tokenizer


DEFAULT_TRAIN_BATCH_SIZE = 16
TARGET_READ_PERCENTS = [25, 50, 75]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def get_text_config(config):
    return getattr(config, "text_config", config)


def full_attention_layers(text_cfg) -> list[int]:
    return [
        idx for idx, layer_type in enumerate(text_cfg.layer_types) if layer_type == "full_attention"
    ]


def nearest_full_attention_layers(text_cfg, target_percents: list[int]) -> list[int]:
    num_layers = text_cfg.num_hidden_layers
    full_layers = full_attention_layers(text_cfg)
    chosen = []
    for percent in target_percents:
        target_layer = num_layers * (percent / 100)
        layer = min(full_layers, key=lambda l: (abs(l - target_layer), l))
        chosen.append(layer)
    return chosen


def integer_percents_for_layers(
    num_layers: int, layers: list[int], reference_percents: list[int]
) -> list[int]:
    resolved = []
    for layer, reference_percent in zip(layers, reference_percents, strict=True):
        candidates = [
            percent
            for percent in range(1, 100)
            if int(num_layers * (percent / 100)) == layer
        ]
        if not candidates:
            raise ValueError(f"Could not find integer percent mapping to layer {layer}")
        best = min(candidates, key=lambda p: (abs(p - reference_percent), p))
        resolved.append(best)
    return resolved


def default_read_layer_percents(text_cfg) -> list[int]:
    layers = nearest_full_attention_layers(text_cfg, TARGET_READ_PERCENTS)
    return integer_percents_for_layers(
        text_cfg.num_hidden_layers, layers, TARGET_READ_PERCENTS
    )


def default_hf_repo_name(model_name: str) -> str:
    return f"activation-oracle-{model_name.split('/')[-1]}"


def first_full_attention_layer(text_cfg) -> int:
    return full_attention_layers(text_cfg)[0]


def build_classification_datasets(main_train_size: int, main_test_size: int) -> dict:
    return {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
        },
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            "batch_size": 4,
        },
        "singular_plural": {
            "num_train": 0,
            "num_test": main_test_size,
            "splits": ["test"],
        },
    }


def apply_defaults(args: argparse.Namespace, text_cfg) -> None:
    if args.layer_percents is None:
        args.layer_percents = ",".join(map(str, default_read_layer_percents(text_cfg)))
    if args.hook_onto_layer is None:
        args.hook_onto_layer = first_full_attention_layer(text_cfg)
    if args.train_batch_size is None:
        args.train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    if args.activation_collection_batch_mult is None:
        args.activation_collection_batch_mult = 8
    if args.eval_batch_mult is None:
        args.eval_batch_mult = 8
    if args.main_train_size is None:
        args.main_train_size = 6000
    if args.main_test_size is None:
        args.main_test_size = 250
    if args.num_epochs is None:
        args.num_epochs = 1
    if args.lr is None:
        args.lr = 1e-5
    if args.lora_r is None:
        args.lora_r = 64
    if args.lora_alpha is None:
        args.lora_alpha = 128
    if args.lora_dropout is None:
        args.lora_dropout = 0.05
    if args.eval_steps is None:
        args.eval_steps = 9_999_999
    if args.save_steps is None:
        args.save_steps = 5000
    if args.window_mult is None:
        args.window_mult = 20
    if args.seed is None:
        args.seed = 42
    if args.save_dir is None:
        args.save_dir = "checkpoints_gemma4_custom"
    if args.wandb_suffix is None:
        args.wandb_suffix = "_gemma4_custom"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        required=True,
        choices=[
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
            "google/gemma-4-26B-A4B-it",
            "google/gemma-4-31B-it",
        ],
    )
    ap.add_argument("--hook-onto-layer", type=int, default=None)
    ap.add_argument("--layer-percents", default=None)
    ap.add_argument("--train-batch-size", type=int, default=None)
    ap.add_argument("--activation-collection-batch-mult", type=int, default=None)
    ap.add_argument("--eval-batch-mult", type=int, default=None)
    ap.add_argument("--main-train-size", type=int, default=None)
    ap.add_argument("--main-test-size", type=int, default=None)
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=None)
    ap.add_argument("--eval-steps", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=None)
    ap.add_argument("--window-mult", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--wandb-project", default="sae_introspection")
    ap.add_argument("--wandb-suffix", default=None)
    ap.add_argument("--hf-repo-name", default=None)
    ap.add_argument("--hf-push-to-hub", action="store_true")
    ap.add_argument("--hf-private-repo", action="store_true")
    ap.add_argument("--skip-if-hub-exists", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=False)
    ap.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    args = parse_args()
    model_name = args.model_name
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    config = AutoConfig.from_pretrained(model_name)
    text_cfg = get_text_config(config)
    apply_defaults(args, text_cfg)
    layer_percents = parse_int_csv(args.layer_percents)
    target_full_layers = nearest_full_attention_layers(text_cfg, TARGET_READ_PERCENTS)
    ple_size = getattr(text_cfg, "hidden_size_per_layer_input", 0)
    kv_shared = getattr(text_cfg, "num_kv_shared_layers", 0)
    full_layer = first_full_attention_layer(text_cfg)
    hook_onto_layer = args.hook_onto_layer
    train_batch_size = args.train_batch_size
    hf_repo_name = args.hf_repo_name or default_hf_repo_name(model_name)
    lora_targets = get_text_only_lora_targets(model_name)

    assert lora_targets is not None, (
        f"No Gemma 4 LoRA targets resolved for {model_name}"
    )
    assert all(0 < p < 100 for p in layer_percents), (
        "layer percents must be in (0, 100)"
    )

    if local_rank == 0:
        print("Model:", model_name)
        print("Text layers:", text_cfg.num_hidden_layers)
        print("Hidden size:", text_cfg.hidden_size)
        print("Sliding window:", text_cfg.sliding_window)
        print("Shared KV layers:", kv_shared)
        print("PLE size:", ple_size)
        print("First full-attention layer:", full_layer)
        print("Hook layer:", hook_onto_layer)
        print(
            "Nearest full-attention read layers for 25/50/75:", target_full_layers
        )
        print("Read layer percents:", layer_percents)
        print("Global train batch size:", train_batch_size)
        print("LoRA target regex:", lora_targets)

    assert train_batch_size % world_size == 0, (
        f"Global batch size {train_batch_size} must be divisible by world_size {world_size}"
    )
    per_rank_train_batch_size = train_batch_size // world_size

    if local_rank == 0:
        print(
            f"Per-rank train batch size: {per_rank_train_batch_size}, world size: {world_size}"
        )

    if args.skip_if_hub_exists and args.hf_push_to_hub:
        try:
            hf_user = hf_whoami()
            hf_owner = hf_user.get("name") if isinstance(hf_user, dict) else None
        except Exception:
            hf_owner = None
        hf_full_repo = f"{hf_owner}/{hf_repo_name}" if hf_owner else hf_repo_name
        if local_rank == 0 and repo_exists(hf_full_repo):
            print(f"Skipping {model_name} because {hf_full_repo} already exists")
            dist.barrier()
            return

    classification_datasets = build_classification_datasets(
        main_train_size=args.main_train_size,
        main_test_size=args.main_test_size,
    )

    # build_loader_groups in nl_probes.sft still reads the module global
    # train_batch_size. Set it explicitly so the shared helper behaves the same
    # way here as it does in the original entrypoint.
    sft_mod.train_batch_size = per_rank_train_batch_size
    loader_groups = sft_mod.build_loader_groups(
        model_name=model_name,
        layer_percents=layer_percents,
        act_collection_batch_size=per_rank_train_batch_size,
        save_acts=False,
        classification_datasets=classification_datasets,
        model_kwargs={},
    )

    dataset_loaders = (
        loader_groups["latentqa_loaders"]
        + loader_groups["classification_loaders"]
        + loader_groups["past_lens_loaders"]
    )
    loader_counts = {
        "latentqa": len(loader_groups["latentqa_loaders"]),
        "classification": len(loader_groups["classification_loaders"]),
        "past_lens": len(loader_groups["past_lens_loaders"]),
    }

    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        hook_onto_layer=hook_onto_layer,
        hf_repo_name=hf_repo_name,
        hf_push_to_hub=args.hf_push_to_hub,
        hf_private_repo=args.hf_private_repo,
        layer_percents=layer_percents,
        train_batch_size=per_rank_train_batch_size,
        activation_collection_batch_size=per_rank_train_batch_size
        * args.activation_collection_batch_mult,
        eval_batch_size=per_rank_train_batch_size * args.eval_batch_mult,
        eval_steps=args.eval_steps,
        eval_on_start=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=1,
        num_epochs=args.num_epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_suffix=args.wandb_suffix,
        window_mult=args.window_mult,
    )
    cfg.finalize(dataset_loaders=dataset_loaders)

    if local_rank == 0:
        print("Resolved act layers:", cfg.act_layers)
        print("Save dir:", cfg.save_dir)
        print("Loader counts:", loader_counts)

    tokenizer = load_tokenizer(cfg.model_name)

    if local_rank == 0:
        sft_mod._ensure_datasets_exist(dataset_loaders)
    dist.barrier()

    all_training_data, all_eval_data = sft_mod.build_datasets(
        cfg,
        dataset_loaders=dataset_loaders,
        window_mult=cfg.window_mult,
    )

    if local_rank == 0:
        print(
            f"training data length: {len(all_training_data)}, eval datasets: {len(all_eval_data)}"
        )

    if args.dry_run:
        if local_rank == 0:
            print("Dry run complete; not starting training.")
        dist.barrier()
        return

    sft_mod.train_model(
        cfg=cfg,
        training_data=all_training_data,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs={},
        verbose=True,
    )


if __name__ == "__main__":
    main()
