from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "reports" / "figures"


def save_current(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {output}")


def plot_rfm_distributions() -> None:
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_features.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        sns.histplot(rfm[col], bins=40, ax=ax, color="#4C78A8")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Customers")

    save_current("01_rfm_distributions.png")


def plot_kmeans_evaluation() -> None:
    evaluation = pd.read_csv(PROCESSED_DIR / "kmeans_evaluation.csv")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(evaluation["K"], evaluation["Inertia"], marker="o", color="#4C78A8", label="Inertia")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(evaluation["K"], evaluation["Silhouette"], marker="s", color="#F58518", label="Silhouette")
    ax2.set_ylabel("Silhouette")
    ax2.tick_params(axis="y")

    plt.title("KMeans Evaluation")
    save_current("02_kmeans_evaluation.png")


def plot_segment_size() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "segment_summary.csv")

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=summary, x="Segment", y="Customers", color="#54A24B")
    ax.set_title("Customer Count by Segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Customers")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")

    save_current("03_segment_size.png")


def plot_segment_rfm() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "segment_summary.csv")
    cols = ["AvgRecency", "AvgFrequency", "AvgMonetary"]
    normalized = summary[["Segment"] + cols].copy()
    for col in cols:
        max_value = normalized[col].max()
        normalized[col] = normalized[col] / max_value

    long = normalized.melt(id_vars="Segment", value_vars=cols, var_name="Metric", value_name="NormalizedValue")

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=long, x="Segment", y="NormalizedValue", hue="Metric")
    ax.set_title("Normalized RFM Profile by Segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Normalized Value")
    ax.legend(title="")

    save_current("04_segment_rfm_profile.png")


def plot_top_rules() -> None:
    rules = pd.read_csv(PROCESSED_DIR / "association_rules.csv")
    top_rules = rules.sort_values("Lift", ascending=False).head(10).copy()
    top_rules["Rule"] = (
        top_rules["Antecedent"].astype(str)
        + " -> "
        + top_rules["Consequent"].astype(str)
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=top_rules, y="Rule", x="Lift", color="#E45756")
    ax.set_title("Top 10 Association Rules by Lift")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Rule")

    save_current("05_top_rules_lift.png")


def main() -> None:
    plot_rfm_distributions()
    plot_kmeans_evaluation()
    plot_segment_size()
    plot_segment_rfm()
    plot_top_rules()


if __name__ == "__main__":
    main()
