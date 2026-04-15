# EvilScript Gemma 4 Report

This repository has been adapted and used to train Activation Oracle LoRA adapters for the Gemma 4 family under the [`EvilScript`](https://huggingface.co/EvilScript) Hugging Face account.

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
