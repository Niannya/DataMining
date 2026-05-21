from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def find_data_file() -> Path:
    candidates = []
    for pattern in ("*.xlsx", "*.xls", "*.zip"):
        candidates.extend(RAW_DIR.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "未在 FinalProject/data/raw/ 中找到 Online Retail II 数据文件。"
        )

    return sorted(candidates)[0]


def extract_excel_from_zip(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        excel_names = [
            name for name in zf.namelist()
            if name.lower().endswith((".xlsx", ".xls"))
        ]
        if not excel_names:
            raise FileNotFoundError("压缩包中没有找到 Excel 数据文件。")

        target_name = excel_names[0]
        output_path = RAW_DIR / Path(target_name).name
        if not output_path.exists():
            zf.extract(target_name, RAW_DIR)
            extracted = RAW_DIR / target_name
            if extracted != output_path:
                extracted.replace(output_path)

    return output_path


def load_online_retail(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        path = extract_excel_from_zip(path)

    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Customer ID": "CustomerID",
        "CustomerID": "CustomerID",
        "InvoiceDate": "InvoiceDate",
        "Invoice": "Invoice",
        "StockCode": "StockCode",
        "Description": "Description",
        "Quantity": "Quantity",
        "Price": "Price",
        "Country": "Country",
    }
    df = df.rename(columns=rename_map)

    needed = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"数据字段缺失：{missing}")

    df = df.dropna(subset=["CustomerID", "InvoiceDate", "StockCode", "Quantity", "Price"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Quantity", "Price"])
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["Amount"] = df["Quantity"] * df["Price"]

    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    return rfm


def run_pipeline() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data_file = find_data_file()
    raw = load_online_retail(data_file)
    clean = clean_transactions(raw)
    rfm = build_rfm(clean)

    clean.to_csv(PROCESSED_DIR / "clean_transactions.csv", index=False, encoding="utf-8-sig")
    rfm.to_csv(PROCESSED_DIR / "rfm_features.csv", index=False, encoding="utf-8-sig")

    print(f"原始记录数: {len(raw)}")
    print(f"清洗后记录数: {len(clean)}")
    print(f"用户数: {len(rfm)}")
    print(f"输出: {PROCESSED_DIR / 'clean_transactions.csv'}")
    print(f"输出: {PROCESSED_DIR / 'rfm_features.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
