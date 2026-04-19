"""Generate figures for the Gemma 4 technical report from the pulled wandb CSVs.

Outputs PNGs under docs/figures/:
  - train_loss_curves.png         : per-model train-loss curve (log-y, smoothed)
  - eval_accuracy_curves.png      : per-model accuracy-over-steps, faceted by dataset
  - final_eval_heatmap.png        : model × dataset final-accuracy heatmap
  - final_eval_bars.png           : grouped bar chart of final accuracy per dataset
  - eval_trajectory_summary.png   : mean-across-datasets accuracy curve per model
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("docs/data")
FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "gemma-4-E2B-it",
    "gemma-4-E4B-it",
    "gemma-4-26B-A4B-it",
    "gemma-4-31B-it",
]
MODEL_COLORS = {
    "gemma-4-E2B-it": "#4c72b0",
    "gemma-4-E4B-it": "#55a868",
    "gemma-4-26B-A4B-it": "#c44e52",
    "gemma-4-31B-it": "#8172b2",
}


def smooth(x: np.ndarray, k: int = 200) -> np.ndarray:
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    smoothed = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_train_loss() -> None:
    df = pd.read_csv(DATA_DIR / "gemma4_training_history.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("_step")
        loss = sub["train/loss"].to_numpy()
        steps = sub["_step"].to_numpy()
        if len(loss) == 0:
            continue
        ax.plot(
            steps,
            smooth(loss, k=400),
            label=model,
            color=MODEL_COLORS[model],
            lw=1.6,
        )
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss (log, 400-step EMA)")
    ax.set_title("Gemma 4 Activation-Oracle training-loss curves")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "train_loss_curves.png", dpi=150)
    plt.close(fig)


def plot_eval_accuracy_facets() -> None:
    df = pd.read_csv(DATA_DIR / "gemma4_eval_history.csv")
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.0 * cols, 2.3 * rows), sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(rows, cols)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols, i % cols]
        sub_ds = df[df["dataset"] == ds]
        for model in MODEL_ORDER:
            sub = sub_ds[sub_ds["model"] == model].sort_values("_step")
            if sub.empty:
                continue
            ax.plot(
                sub["_step"].to_numpy(),
                sub["ans_correct"].to_numpy(),
                color=MODEL_COLORS[model],
                lw=1.4,
                marker="o",
                markersize=2.5,
                label=model,
            )
        ax.set_title(ds, fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_ylim(0.0, 1.0)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    # put legend on the first axis
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.supxlabel("training step")
    fig.supylabel("answer-correct rate")
    fig.suptitle("Eval accuracy during training — by classification dataset", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_accuracy_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_heatmap() -> None:
    with (DATA_DIR / "gemma4_run_summary.json").open() as f:
        summary = json.load(f)
    # build dataframe of final eval scores
    rows = []
    for s in summary:
        model = s["model"]
        for ds, v in s["eval_ans_correct"].items():
            rows.append({"model": model, "dataset": ds, "ans_correct": float(v)})
    piv = (
        pd.DataFrame(rows)
        .pivot(index="model", columns="dataset", values="ans_correct")
        .reindex(MODEL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(0.9 * len(piv.columns) + 1, 0.6 * len(piv) + 1.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.75 else "black",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("answer-correct rate")
    ax.set_title("Final eval accuracy — Gemma 4 activation oracles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_eval_heatmap.png", dpi=150)
    plt.close(fig)
    piv.to_csv(DATA_DIR / "gemma4_final_eval_matrix.csv")


def plot_final_bars() -> None:
    with (DATA_DIR / "gemma4_run_summary.json").open() as f:
        summary = json.load(f)
    datasets = sorted({ds for s in summary for ds in s["eval_ans_correct"]})
    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, model in enumerate(MODEL_ORDER):
        s = next((s for s in summary if s["model"] == model), None)
        if s is None:
            continue
        vals = [s["eval_ans_correct"].get(ds, np.nan) for ds in datasets]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("answer-correct rate")
    ax.set_title("Final evaluation accuracy per dataset — Gemma 4 oracles")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, ncols=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_eval_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_mean() -> None:
    df = pd.read_csv(DATA_DIR / "gemma4_eval_history.csv")
    avg = df.groupby(["model", "_step"], as_index=False)["ans_correct"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for model in MODEL_ORDER:
        sub = avg[avg["model"] == model].sort_values("_step")
        if sub.empty:
            continue
        ax.plot(
            sub["_step"].to_numpy(),
            sub["ans_correct"].to_numpy(),
            lw=1.8,
            marker="o",
            markersize=4,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("mean answer-correct rate (10 datasets)")
    ax.set_title("Cross-dataset mean eval accuracy during training")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_trajectory_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plot_train_loss()
    print("wrote train_loss_curves.png")
    plot_eval_accuracy_facets()
    print("wrote eval_accuracy_curves.png")
    plot_final_heatmap()
    print("wrote final_eval_heatmap.png")
    plot_final_bars()
    print("wrote final_eval_bars.png")
    plot_trajectory_mean()
    print("wrote eval_trajectory_summary.png")


if __name__ == "__main__":
    main()
