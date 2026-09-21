import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data_pipeline" / "data" / "sample_sales.csv"
OUTPUT_DIR = ROOT / "data_pipeline" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [c.strip().lower().replace(" ", "_") for c in cleaned.columns]

    for col in ["date", "region", "product", "channel"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    if "quantity" in cleaned.columns:
        cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce").fillna(0)
        cleaned["quantity"] = cleaned["quantity"].clip(lower=0)

    if "unit_price" in cleaned.columns:
        cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce").fillna(0)
        cleaned["unit_price"] = cleaned["unit_price"].clip(lower=0)

    if "revenue" in cleaned.columns:
        cleaned["revenue"] = pd.to_numeric(cleaned["revenue"], errors="coerce").fillna(0)
        cleaned["revenue"] = cleaned["revenue"].clip(lower=0)
    else:
        cleaned["revenue"] = cleaned["quantity"] * cleaned["unit_price"]

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned = cleaned.dropna(subset=["date"]).copy()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def write_outputs(df: pd.DataFrame) -> dict:
    processed_path = OUTPUT_DIR / "processed_sales.csv"
    summary_path = OUTPUT_DIR / "etl_summary.json"

    df.to_csv(processed_path, index=False)
    summary = {
        "rows_loaded": int(len(df)),
        "regions": sorted(df["region"].unique().tolist()) if "region" in df.columns else [],
        "products": sorted(df["product"].unique().tolist()) if "product" in df.columns else [],
        "total_revenue": round(float(df["revenue"].sum()), 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    df = load_data(RAW_PATH)
    cleaned = clean_data(df)
    summary = write_outputs(cleaned)
    print(f"Pipeline complete. Rows processed: {summary['rows_loaded']}")
    print(f"Total revenue: ${summary['total_revenue']:,.2f}")


if __name__ == "__main__":
    main()
