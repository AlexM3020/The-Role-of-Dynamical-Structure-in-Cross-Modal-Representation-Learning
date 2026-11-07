"""Descriptive plots and world-level paired intervals without significance claims."""

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import paired_bootstrap


def contrast_summary(results: pd.DataFrame, metric: str = "cka") -> pd.DataFrame:
    groups = ["case", "n_pairs", "world", "condition"]
    # Initialization repeats are nested in world; never count them as extra worlds.
    means = results.groupby(groups, dropna=False)[metric].mean().unstack("condition")
    if not {"local", "scrambled"}.issubset(means.columns):
        raise ValueError("Both local and scrambled conditions are required.")
    differences = (means["local"] - means["scrambled"]).rename("difference").reset_index()
    rows = []
    for (case, count), block in differences.groupby(["case", "n_pairs"]):
        rows.append({"case": case, "n_pairs": count, "metric": metric,
                     "contrast": "local - scrambled", **paired_bootstrap(block["difference"])})
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, metric: str = "cka"):
    cases = list(results["case"].unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 3.8), squeeze=False)
    for ax, case in zip(axes[0], cases):
        block = results[results["case"] == case]
        world_means = block.groupby(["condition", "n_pairs", "world"])[metric].mean().reset_index()
        for condition, condition_rows in world_means.groupby("condition"):
            color = {"local": "tab:blue", "scrambled": "tab:orange", "iid": "tab:gray"}.get(condition, "black")
            for _, world in condition_rows.groupby("world"):
                ax.plot(world["n_pairs"], world[metric], color=color, alpha=0.25, linewidth=1)
            averaged = condition_rows.groupby("n_pairs")[metric].mean()
            ax.plot(averaged.index, averaged.values, "o-", color=color, label=condition, linewidth=2)
        ax.set(xscale="log", xlabel="Independent training transitions per encoder",
               ylabel=metric, title=case)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Descriptive means; faint lines are independent worlds")
    fig.tight_layout()
    return fig
