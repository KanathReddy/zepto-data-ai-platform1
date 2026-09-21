import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_pipeline" / "output" / "processed_sales.csv"
OUTPUT_DIR = ROOT / "support_assistant" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def answer_question(df: pd.DataFrame, question: str) -> str:
    q = question.lower()

    region_revenue = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    product_volume = df.groupby("product")["quantity"].sum().sort_values(ascending=False)
    channel_revenue = df.groupby("channel")["revenue"].sum().sort_values(ascending=False)

    if "region" in q and "revenue" in q and "most" in q:
        best_region = region_revenue.idxmax()
        best_value = region_revenue.max()
        return f"The {best_region} region generated the highest revenue, totaling ${best_value:,.2f}."

    if "product" in q and ("sales volume" in q or "volume" in q or "highest" in q):
        best_product = product_volume.idxmax()
        best_value = product_volume.max()
        return f"The {best_product} product had the highest sales volume, with {best_value:,.0f} units sold."

    if "channel" in q and ("best" in q or "highest" in q or "revenue" in q):
        best_channel = channel_revenue.idxmax()
        best_value = channel_revenue.max()
        return f"The {best_channel} channel performed best by revenue, generating ${best_value:,.2f}."

    if "trend" in q or "trends" in q or "sales" in q:
        monthly = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("M").astype(str))
        trend = monthly.groupby("month")["revenue"].sum().sort_values(ascending=False)
        best_month = trend.idxmax()
        best_month_value = trend.max()
        return f"The strongest sales month was {best_month}, which generated ${best_month_value:,.2f}."

    return "I can answer questions about region revenue, product sales volume, channel performance, and sales trends."


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "Which region generated the most revenue?"

    df = load_data(DATA_PATH)
    response = answer_question(df, question)
    print(response)

    record = {"question": question, "answer": response}
    record_path = OUTPUT_DIR / "assistant_log.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
