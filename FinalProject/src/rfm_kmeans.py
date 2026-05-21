from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_rfm() -> pd.DataFrame:
    path = PROCESSED_DIR / "rfm_features.csv"
    if not path.exists():
        raise FileNotFoundError("请先运行 online_retail_pipeline.py 生成 rfm_features.csv")
    return pd.read_csv(path)


def prepare_features(rfm: pd.DataFrame) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # RFM 分布很偏，先做 log 变换，减少极端值影响
    features["Recency"] = features["Recency"].clip(lower=0)
    features["Frequency"] = features["Frequency"].clip(lower=1)
    features["Monetary"] = features["Monetary"].clip(lower=0)
    features = np.log1p(features)

    scaled = StandardScaler().fit_transform(features)
    return pd.DataFrame(scaled, columns=["Recency", "Frequency", "Monetary"])


def evaluate_k(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(features)
        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(features, labels),
        })
    return pd.DataFrame(rows)


def make_cluster_summary(result: pd.DataFrame) -> pd.DataFrame:
    summary = result.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        MedianMonetary=("Monetary", "median"),
    ).reset_index()

    summary["CustomerShare"] = summary["Customers"] / summary["Customers"].sum()
    return summary.sort_values("Segment")


def run_kmeans(k: int = 4) -> None:
    rfm = load_rfm()
    features = prepare_features(rfm)
    evaluation = evaluate_k(features)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    result = rfm.copy()
    result["Segment"] = model.fit_predict(features)
    summary = make_cluster_summary(result)

    evaluation.to_csv(PROCESSED_DIR / "kmeans_evaluation.csv", index=False, encoding="utf-8-sig")
    result.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(PROCESSED_DIR / "segment_summary.csv", index=False, encoding="utf-8-sig")

    print("K 值评估:")
    print(evaluation.round(4).to_string(index=False))
    print("\n分群摘要:")
    print(summary.round(4).to_string(index=False))
    print(f"\n输出: {PROCESSED_DIR / 'kmeans_evaluation.csv'}")
    print(f"输出: {PROCESSED_DIR / 'customer_segments.csv'}")
    print(f"输出: {PROCESSED_DIR / 'segment_summary.csv'}")


if __name__ == "__main__":
    run_kmeans()
