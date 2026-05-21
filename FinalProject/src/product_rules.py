from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_transactions() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean_transactions.csv"
    if not path.exists():
        raise FileNotFoundError("请先运行 online_retail_pipeline.py 生成 clean_transactions.csv")
    return pd.read_csv(path, usecols=["Invoice", "StockCode", "Description"])


def build_baskets(df: pd.DataFrame, max_items: int = 100) -> list[tuple[str, ...]]:
    df = df.copy()
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)

    # 去掉邮费、手续费等非普通商品
    bad_desc = df["Description"].str.contains("POSTAGE|CARRIAGE|MANUAL|DISCOUNT|AMAZON FEE", case=False, na=False)
    df = df[~bad_desc]

    baskets = []
    for _, group in df.groupby("Invoice"):
        items = tuple(sorted(set(group["StockCode"])))
        if 2 <= len(items) <= max_items:
            baskets.append(items)

    return baskets


def make_description_map(df: pd.DataFrame) -> dict[str, str]:
    desc_map = {}
    counts = df.groupby(["StockCode", "Description"]).size().reset_index(name="Count")
    counts = counts.sort_values(["StockCode", "Count"], ascending=[True, False])
    for _, row in counts.drop_duplicates("StockCode").iterrows():
        desc_map[str(row["StockCode"])] = str(row["Description"])
    return desc_map


def mine_pair_rules(baskets: list[tuple[str, ...]], min_support: float = 0.015, min_confidence: float = 0.25) -> pd.DataFrame:
    basket_count = len(baskets)
    item_counts = Counter()
    pair_counts = Counter()

    for basket in baskets:
        item_counts.update(basket)
        pair_counts.update(combinations(basket, 2))

    rows = []
    min_pair_count = max(1, int(basket_count * min_support))
    for (left, right), pair_count in pair_counts.items():
        if pair_count < min_pair_count:
            continue

        support = pair_count / basket_count

        for antecedent, consequent in ((left, right), (right, left)):
            confidence = pair_count / item_counts[antecedent]
            consequent_support = item_counts[consequent] / basket_count
            lift = confidence / consequent_support

            if confidence >= min_confidence:
                rows.append({
                    "Antecedent": antecedent,
                    "Consequent": consequent,
                    "Support": support,
                    "Confidence": confidence,
                    "Lift": lift,
                    "Count": pair_count,
                })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    return rules.sort_values(["Lift", "Confidence", "Support"], ascending=False).reset_index(drop=True)


def run_rules() -> None:
    df = load_transactions()
    baskets = build_baskets(df)
    desc_map = make_description_map(df)
    rules = mine_pair_rules(baskets)

    if not rules.empty:
        rules["AntecedentName"] = rules["Antecedent"].map(desc_map)
        rules["ConsequentName"] = rules["Consequent"].map(desc_map)

    rules.to_csv(PROCESSED_DIR / "association_rules.csv", index=False, encoding="utf-8-sig")

    print(f"用于关联规则的订单数: {len(baskets)}")
    print(f"规则数量: {len(rules)}")
    if not rules.empty:
        cols = ["Antecedent", "AntecedentName", "Consequent", "ConsequentName", "Support", "Confidence", "Lift", "Count"]
        print(rules[cols].head(20).round(4).to_string(index=False))
    print(f"输出: {PROCESSED_DIR / 'association_rules.csv'}")


if __name__ == "__main__":
    run_rules()
