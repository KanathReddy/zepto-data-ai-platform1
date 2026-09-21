import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data_pipeline" / "output" / "processed_sales.csv"
OUTPUT_DIR = ROOT / "analytics" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_summary(df: pd.DataFrame) -> dict:
    summary = {
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "average_transaction_value": round(float(df["revenue"].mean()), 2),
        "total_orders": int(len(df)),
        "top_region": df.groupby("region")["revenue"].sum().sort_values(ascending=False).idxmax(),
        "top_product": df.groupby("product")["quantity"].sum().sort_values(ascending=False).idxmax(),
        "revenue_by_region": df.groupby("region")["revenue"].sum().sort_values(ascending=False).to_dict(),
    }
    return summary


def save_chart(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Revenue by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    df = load_data(PROCESSED_PATH)
    summary = compute_summary(df)
    save_chart(df, OUTPUT_DIR / "revenue_by_region.png")

    summary_path = OUTPUT_DIR / "analytics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Total revenue: ${summary['total_revenue']:,}")
    print(f"Top region: {summary['top_region']}")
    print(f"Top product by volume: {summary['top_product']}")


if __name__ == "__main__":
    main()
