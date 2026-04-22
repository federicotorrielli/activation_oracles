# Gemma 4 Oracle Training Notes

This file collects the Gemma 4 facts most relevant to training activation-oracle style verbalizers in this repo, plus the design choices encoded in `nl_probes/gemma4_sft.py`.

## Model Family Summary

All four public Gemma 4 instruction models are exposed through `Gemma4ForConditionalGeneration`, even for text-only use. That matters for this repo because LoRA targeting, hidden-state hooks, and tokenization all have to go through the multimodal wrapper instead of assuming a plain decoder-only path.

| Model | Core type | Audio | Context | Layers | Hidden size | Sliding window | Shared KV | PLE | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `google/gemma-4-E2B-it` | dense | yes | 128k | 35 | 1536 | 512 | 20 | yes (`256`) | Small dense model with per-layer embeddings and shared-KV upper stack |
| `google/gemma-4-E4B-it` | dense | yes | 128k | 42 | 2560 | 512 | 18 | yes (`256`) | Same architectural regime as E2B, larger hidden size |
| `google/gemma-4-26B-A4B-it` | MoE | no | 256k | 30 | 2816 | 1024 | 0 | no | 128 experts, top-8 routing, attention-only LoRA is the sane default |
| `google/gemma-4-31B-it` | dense | no | 256k | 60 | 5376 | 1024 | 0 | no | Largest and cleanest hidden-state transport target |

## Architecture Details That Matter For Oracles

### 1. Multimodal wrapper

- All Gemma 4 models sit under a conditional-generation wrapper.
- In this repo, the text tower lives at `model.model.language_model.layers[...]`.
- PEFT-wrapped Gemma 4 models add one more `base_model.model.model.language_model.layers[...]` level.

This is already patched in the main repo, but any new trainer must assume that path.

### 2. Per-layer embeddings (PLE)

- Present on `E2B` and `E4B`.
- Exposed in config as `hidden_size_per_layer_input=256`.
- Each decoder layer receives a token-conditioned side input in addition to the normal residual stream.

Implication: transporting only the residual hidden state is less complete on `E2B/E4B` than on more standard decoder stacks.

### 3. Shared KV cache

- Present on `E2B` and `E4B`.
- `E2B`: `num_kv_shared_layers=20`
- `E4B`: `num_kv_shared_layers=18`
- Absent on `26B-A4B` and `31B`.

Implication: later-layer attention state on the small dense models is less independent than the repo's original Qwen/Gemma2 assumptions.

### 4. Alternating sliding and full attention

- `E2B`: four sliding layers, then one full-attention layer, repeating.
- `E4B`, `26B-A4B`, `31B`: five sliding layers, then one full-attention layer, repeating.
- The first full-attention layer is therefore:
  - `E2B`: layer `4`
  - `E4B`: layer `5`
  - `26B-A4B`: layer `5`
  - `31B`: layer `5`

Implication: if we want a Gemma-specific injection layer, the first full-attention layer is a better default than the old generic `hook_onto_layer=1`.

### 5. LoRA targeting

The current repo patch uses LM-only regex targeting:

- Dense Gemma 4:
  - `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`
- `26B-A4B` MoE:
  - `model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj)`

This avoids vision/audio modules and prevents MoE expert-adapter explosion on `26B-A4B`.

## How The Released Gemma 4 Verbalizers Were Trained

The released Gemma 4 verbalizers were trained by `nl_probes/sft.py` with:

- data mixture: `LatentQA + Classification + PastLens`
- read layers: `25/50/75%`
- injection layer: `1`
- LoRA: `r=64`, `alpha=128`, dropout `0.05`
- one epoch over ~1.02M examples

This setup was enough to produce healthy classification-style eval curves, but it did not establish that the oracle verbalizer works for the taboo extraction pipeline in `probabilistic_activation_oracles`.

## Why A Gemma-4-Specific Trainer Exists

The current evidence suggests that Gemma 4 needs an architecture-aware entrypoint, even when the training recipe itself stays close to the original.

The custom trainer in `nl_probes/gemma4_sft.py` differs from `nl_probes/sft.py` only where Gemma 4 forces it to:

1. It only accepts the four Gemma 4 instruction models.
2. It inspects the text config before training (layers, hidden size, sliding window, shared-KV count, PLE size, full-attention layer indices).
3. It defaults the injection layer to the first full-attention layer of each model, not the generic `layer=1`.
4. It picks read-layer percents that map to actual full-attention layers nearest 25/50/75% of depth, instead of using the raw 25/50/75 indices that may land on sliding-window layers.
5. It uses `get_text_only_lora_targets(model_name)` so the LoRA regex stays inside the language tower (and stays attention-only on the `26B-A4B` MoE) instead of leaking into vision/audio/expert modules.

The training mixture, batch size, LoRA rank, learning rate, optimizer, and number of epochs all match the original Qwen3-8B recipe (see "Current Defaults" below).

## Current Defaults (`nl_probes/gemma4_sft.py`)

The trainer now exposes a single set of defaults — the previous `qwen_exact` / `gemma_custom` recipe split has been removed.

- Global train batch size: `16` for every Gemma 4 model.
- Mixture (1× each, no reweighting), exactly matching `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B/ao_config.json`:
  - `LatentQA`: 1 loader, 100k examples
  - `PastLens`: 2 loaders (single-token + multi-token variants), 100k each
  - `Classification`: 2 loaders per dataset (single-token and multi-token QA variants), 6000 train / 250 test per dataset
- Classification datasets: `geometry_of_truth`, `relations`, `sst2`, `md_gender`, `snli`, `ag_news`, `ner`, `tense`, `language_identification`, `singular_plural` (test-only for `singular_plural`; `language_identification` uses `batch_size=4` to fit its long sequences).
- Read layers: nearest full-attention layers to 25/50/75% of depth, expressed as integer percents.
- Injection layer: first full-attention layer of the model.
- LoRA: `r=64`, `alpha=128`, dropout `0.05`, target regex from `get_text_only_lora_targets`.
- Optimization: 1 epoch, `lr=1e-5`, `gradient_accumulation_steps=1`, `gradient_checkpointing=False` (enable with `--gradient-checkpointing` if memory pressure forces it).
- Eval / save cadence: mid-train eval effectively disabled (`eval_steps=9_999_999`, `eval_on_start=False`); checkpoints every 5000 optimizer steps.
- Activation collection batch = `train_batch_size * 8` = 128; eval batch = `train_batch_size * 8` = 128.
- Bookkeeping: `save_dir=checkpoints_gemma4_custom`, `wandb_suffix=_gemma4_custom`, `hf_repo_name=activation-oracle-<model-suffix>`, `seed=42`, `window_mult=20`.

These defaults are not proven optimal yet, but they keep the recipe close to the released Qwen3-8B oracle while staying honest about Gemma 4's architecture.

## Practical Recommendations

### Best hidden-state transport targets

1. `google/gemma-4-31B-it`
2. `google/gemma-4-26B-A4B-it`
3. `google/gemma-4-E4B-it`
4. `google/gemma-4-E2B-it`

Reason: `31B` and `26B-A4B` do not have PLE or shared-KV, so the hidden state is architecturally closer to the original oracle assumptions.

### Best LoRA strategy per model

- `E2B`, `E4B`, `31B`: LM attention + MLP
- `26B-A4B`: attention-only

### If a future retrain still fails

The next move should be changing the transport mechanism itself, not just the hyperparameters:

- learned projector before injection
- multi-layer transport instead of single hidden-state injection
- Gemma-specific validation on taboo extraction during training, not only classification eval

## Sources

- Internal port report: `docs/evilscript_gemma4_report.md`
- Generic oracle trainer: `nl_probes/sft.py`
- LoRA target selection: `nl_probes/utils/activation_utils.py`
- Hugging Face Gemma 4 blog and model configs on the Hub
