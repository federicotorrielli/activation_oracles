"""Pull training loss and eval metrics from the 4 Gemma 4 SFT runs on wandb.

Writes two CSVs under docs/data/:
  - gemma4_training_history.csv   (per-step train loss / LR)
  - gemma4_eval_history.csv       (per-step eval accuracy per dataset, per model)

and a JSON with run-level summary stats for direct quoting in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import wandb

PROJECT = "diunito/sae_introspection"
OUT_DATA = Path("docs/data")
OUT_DATA.mkdir(parents=True, exist_ok=True)


def short_name(run_name: str) -> str:
    for key in ("E2B-it", "E4B-it", "26B-A4B-it", "31B-it"):
        if key in run_name:
            return f"gemma-4-{key}"
    return run_name


def main() -> None:
    api = wandb.Api()
    runs = list(api.runs(PROJECT))

    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    CLASSIFICATION_DATASETS = [
        "ag_news",
        "geometry_of_truth",
        "language_identification",
        "md_gender",
        "ner",
        "relations",
        "singular_plural",
        "snli",
        "sst2",
        "tense",
    ]

    for r in runs:
        model = short_name(r.name)
        print(f"Pulling {model} (id={r.id}) …")

        # --- training loss: one series ---
        train_keys = ["_step", "train/loss", "train/learning_rate"]
        train_rows = list(r.scan_history(keys=train_keys))
        if train_rows:
            t = pd.DataFrame(train_rows)
            t["model"] = model
            t["run_id"] = r.id
            train_frames.append(t)
            print(f"  train rows: {len(t)}")

        # --- eval metrics: fetched per dataset via explicit keys ---
        for ds in CLASSIFICATION_DATASETS:
            ans_key = f"eval_ans_correct/classification_{ds}"
            fmt_key = f"eval_format_correct/classification_{ds}"
            rows = list(r.scan_history(keys=["_step", ans_key, fmt_key]))
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["model"] = model
            df["run_id"] = r.id
            df["dataset"] = ds
            df = df.rename(
                columns={ans_key: "ans_correct", fmt_key: "format_correct"}
            )
            eval_frames.append(df)
        print(f"  eval datasets pulled: {len(CLASSIFICATION_DATASETS)}")

        summaries.append(
            {
                "model": model,
                "run_id": r.id,
                "state": r.state,
                "runtime_s": r.summary.get("_runtime"),
                "total_steps": r.summary.get("_step"),
                "tokens_per_epoch": r.summary.get("train/tokens_per_epoch_est"),
                "num_examples_pre_shard": r.summary.get("train/num_examples_pre_shard"),
                "final_train_loss": r.summary.get("train/loss"),
                "eval_ans_correct": {
                    k.replace("eval_ans_correct/classification_", ""): v
                    for k, v in r.summary.items()
                    if k.startswith("eval_ans_correct/")
                },
                "eval_format_correct": {
                    k.replace("eval_format_correct/classification_", ""): v
                    for k, v in r.summary.items()
                    if k.startswith("eval_format_correct/")
                },
            }
        )

    if train_frames:
        train_df = pd.concat(train_frames, ignore_index=True)
        train_df.to_csv(OUT_DATA / "gemma4_training_history.csv", index=False)
        print(f"Wrote {OUT_DATA/'gemma4_training_history.csv'} "
              f"({len(train_df)} rows)")

    if eval_frames:
        eval_df = pd.concat(eval_frames, ignore_index=True)
        eval_df.to_csv(OUT_DATA / "gemma4_eval_history.csv", index=False)
        print(f"Wrote {OUT_DATA/'gemma4_eval_history.csv'} "
              f"({len(eval_df)} rows)")

    with (OUT_DATA / "gemma4_run_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"Wrote {OUT_DATA/'gemma4_run_summary.json'}")


if __name__ == "__main__":
    main()
