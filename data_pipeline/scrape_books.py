from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com"
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data_pipeline" / "books.db"
CSV_PATH = ROOT / "data_pipeline" / "clean_books.csv"
OUTPUT_PATH = ROOT / "data_pipeline" / "query_outputs.json"
FIXED_GBP_TO_INR = 105.50


def fetch_category_urls() -> list[tuple[str, str]]:
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for link in soup.select("ul.nav-list li a"):
        href = link.get("href", "")
        if "/category/" in href:
            clean_href = urljoin(BASE_URL, href)
            cat_name = link.get_text(strip=True)
            urls.append((cat_name, clean_href))
    unique = []
    seen = set()
    for name, href in urls:
        if href not in seen:
            seen.add(href)
            unique.append((name, href))
    return unique[:6]


def parse_book_card(card: Any, category_name: str) -> dict[str, Any]:
    title = card.select_one("h3 a")
    title_text = title.get("title", "").strip() if title else ""

    price_text = card.select_one("p.price_color")
    price_value = price_text.get_text(strip=True) if price_text else "0"
    cleaned_price = re.sub(r"[^0-9.\-]", "", price_value)

    rating_elem = card.select_one("p.star-rating")
    rating_class = rating_elem["class"][1] if rating_elem and len(rating_elem["class"]) > 1 else ""
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    rating = rating_map.get(rating_class, 0)

    availability_text = card.select_one("p.availability")
    availability = availability_text.get_text(" ", strip=True) if availability_text else ""
    in_stock = "in stock" in availability.lower()

    return {
        "title": title_text,
        "price_gbp": float(cleaned_price) if cleaned_price else 0.0,
        "rating": int(rating),
        "in_stock": in_stock,
        "category": category_name,
    }


def scrape_books() -> pd.DataFrame:
    category_rows = fetch_category_urls()
    records = []
    seen = 0
    required = 60

    for category_name, category_url in category_rows:
        page_num = 1
        while page_num <= 2 and seen < required:
            if page_num == 1:
                page_url = category_url
            else:
                page_url = category_url.replace("/index.html", f"/page-{page_num}.html")
            try:
                response = requests.get(page_url, timeout=30)
                response.raise_for_status()
            except requests.RequestException:
                break
            soup = BeautifulSoup(response.text, "html.parser")
            cards = soup.select("article.product_pod")
            if not cards:
                break
            for card in cards:
                record = parse_book_card(card, category_name)
                if record["title"]:
                    records.append(record)
                    seen += 1
                    if seen >= required:
                        break
            page_num += 1
        if seen >= required:
            break

    if len(records) < 60:
        raise RuntimeError(f"Expected at least 60 books, but only found {len(records)}.")

    df = pd.DataFrame(records[:required])
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce").fillna(df["price_gbp"].median())
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(round(df["rating"].median()))
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    df["in_stock"] = df["in_stock"].astype(bool)
    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]
    return df


def create_database(df: pd.DataFrame) -> pd.DataFrame:
    conn = __import__("sqlite3").connect(DB_PATH)
    categories_df = pd.DataFrame({"category_name": sorted(df["category"].unique().tolist())})
    categories_df["category_id"] = range(1, len(categories_df) + 1)
    categories_df = categories_df[["category_id", "category_name"]]

    books_df = df.merge(categories_df, left_on="category", right_on="category_name", how="left")
    books_df = books_df[[
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id",
    ]].copy()
    books_df["in_stock"] = books_df["in_stock"].astype(int)

    categories_df.to_sql("categories", conn, if_exists="replace", index=False)
    books_df.to_sql("books", conn, if_exists="replace", index=False)

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_categories_name ON categories(category_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_books_category ON books(category_id)")
    conn.commit()
    conn.close()
    return books_df


def run_queries() -> dict[str, Any]:
    conn = __import__("sqlite3").connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM books", conn)
    categories = pd.read_sql("SELECT * FROM categories", conn)

    queries = {
        "select_where": "SELECT title, price_inr FROM books WHERE in_stock = 1 AND rating >= 4 ORDER BY price_inr DESC LIMIT 10;",
        "order_by_limit": "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10;",
        "distinct": "SELECT DISTINCT rating FROM books ORDER BY rating;",
        "between": "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 500 AND 1000 ORDER BY price_inr;",
        "join": "SELECT b.title, c.category_name, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;",
    }

    results = {}
    for name, query in queries.items():
        results[name] = pd.read_sql(query, conn).to_dict(orient="records")

    conn.close()

    query1_df = pd.read_sql(queries["select_where"], __import__("sqlite3").connect(DB_PATH))
    query2_df = pd.read_sql(queries["join"], __import__("sqlite3").connect(DB_PATH))
    in_memory = df.merge(categories, on="category_id", how="inner")[["title", "category_name", "rating", "price_inr"]].sort_values(["rating", "price_inr"], ascending=[False, False]).head(10).reset_index(drop=True)
    query2_df = query2_df.reset_index(drop=True)
    assert query2_df.equals(in_memory), "Join query and pandas merge outputs do not match."
    return results


def main() -> None:
    df = scrape_books()
    df.to_csv(CSV_PATH, index=False)
    create_database(df)
    output = run_queries()
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Scraped and cleaned {len(df)} books.")
    print(f"Saved database to {DB_PATH}")
    print(f"Saved query outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
