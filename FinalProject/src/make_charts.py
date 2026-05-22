from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "reports" / "figures"


def setup_chinese_font() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False


def save_current(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {output}")


def plot_rfm_distributions() -> None:
    rfm = pd.read_csv(PROCESSED_DIR / "rfm_features.csv")
    label_map = {
        "Recency": "log(最近购买间隔+1)",
        "Frequency": "log(购买频次+1)",
        "Monetary": "log(消费金额+1)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        sns.histplot(np.log1p(rfm[col]), bins=40, ax=ax, color="#4C78A8")
        ax.set_title(f"{label_map[col]}分布")
        ax.set_xlabel(label_map[col])
        ax.set_ylabel("客户数")

    save_current("01_rfm_distributions.png")


def plot_kmeans_evaluation() -> None:
    evaluation = pd.read_csv(PROCESSED_DIR / "kmeans_evaluation.csv")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(evaluation["K"], evaluation["Inertia"], marker="o", color="#4C78A8", label="簇内误差平方和")
    ax1.set_xlabel("聚类数 K")
    ax1.set_ylabel("簇内误差平方和")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(evaluation["K"], evaluation["Silhouette"], marker="s", color="#F58518", label="轮廓系数")
    ax2.set_ylabel("轮廓系数")
    ax2.tick_params(axis="y")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title("KMeans 聚类效果评估")
    save_current("02_kmeans_evaluation.png")


def plot_segment_size() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "segment_summary.csv")

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=summary, x="Segment", y="Customers", color="#54A24B")
    ax.set_title("各用户分群人数")
    ax.set_xlabel("用户分群")
    ax.set_ylabel("客户数")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")

    save_current("03_segment_size.png")


def plot_segment_rfm() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "segment_summary.csv")
    cols = ["AvgRecency", "AvgFrequency", "AvgMonetary"]
    metric_map = {
        "AvgRecency": "平均最近购买间隔",
        "AvgFrequency": "平均购买频次",
        "AvgMonetary": "平均消费金额",
    }
    normalized = summary[["Segment"] + cols].copy()
    for col in cols:
        max_value = normalized[col].max()
        normalized[col] = normalized[col] / max_value

    long = normalized.melt(id_vars="Segment", value_vars=cols, var_name="Metric", value_name="NormalizedValue")
    long["Metric"] = long["Metric"].map(metric_map)

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=long, x="Segment", y="NormalizedValue", hue="Metric")
    ax.set_title("各用户分群标准化 RFM 画像")
    ax.set_xlabel("用户分群")
    ax.set_ylabel("标准化数值")
    ax.legend(title="指标")

    save_current("04_segment_rfm_profile.png")


def plot_top_rules() -> None:
    rules = pd.read_csv(PROCESSED_DIR / "association_rules.csv")
    top_rules = rules.sort_values("Lift", ascending=False).head(10).copy()
    top_rules["Rule"] = (
        "商品"
        + top_rules["Antecedent"].astype(str)
        + " → 商品"
        + top_rules["Consequent"].astype(str)
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=top_rules, y="Rule", x="Lift", color="#E45756")
    ax.set_title("提升度排名前 10 的商品关联规则")
    ax.set_xlabel("提升度")
    ax.set_ylabel("关联规则")

    save_current("05_top_rules_lift.png")


def main() -> None:
    setup_chinese_font()
    plot_rfm_distributions()
    plot_kmeans_evaluation()
    plot_segment_size()
    plot_segment_rfm()
    plot_top_rules()


if __name__ == "__main__":
    main()
