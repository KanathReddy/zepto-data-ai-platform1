# Data Pipeline Module

## Goal

This module scrapes public catalog data from `books.toscrape.com`, cleans it defensibly, converts the package price to INR using the project-defined baseline rate, and loads it into a normalized SQLite database.

## Fixed conversion rate

The required baseline conversion rate is exactly:

1 GBP = 105.50 INR

This is an artificial project constant with no live lookup or date, and it is the basis for the `price_inr` field.

## Data cleaning decisions

The pipeline uses a median-imputation strategy for numeric fields when parsing fails, because it is robust to outliers and keeps the dataset usable without discarding a large portion of books. Rows with missing identifiers such as title or category are dropped because those values are essential for a valid relational record. Availability parsing failures are defaulted to `False` so the boolean `in_stock` field remains safe and interpretable.

## Run

From the repository root:

```bash
python data_pipeline/scrape_books.py
```

This regenerates the SQLite database and all SQL query outputs.

## Output artifacts

- `data_pipeline/books.db` — normalized SQLite database with `categories` and `books` tables.
- `data_pipeline/clean_books.csv` — cleaned dataset exported as CSV.
- `data_pipeline/query_outputs.json` — saved SQL query strings and their output records.

## Schema design

The database uses the required two-table relational model:

- `categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)`
- `books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))`

## Example SQL coverage

The script executes at least five queries covering:

- `SELECT` + `WHERE`
- `ORDER BY`
- `LIMIT`
- `DISTINCT`
- `BETWEEN`
- `JOIN`

It also reads selected query results with `pd.read_sql` and compares them with a direct `pd.merge` result to confirm they are equivalent.
