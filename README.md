# Zepto Data AI Platform

This repository is a single public GitHub project containing three modules at the root:

- `data_pipeline/` — a web-scraping, cleaning, conversion, and SQLite loading pipeline for catalog-style books data.
- `analytics/` — a Titanic EDA + modeling workflow with saved offline fallback artifacts and a trained pipeline.
- `support_assistant/` — a local RAG-style support assistant backed by ChromaDB and a LangGraph router with a deterministic mock mode.

The repository uses one consolidated dependency file at the root: `requirements.txt`.

## Project setup

```bash
cd zepto-data-ai-platform
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Module 1: Data Pipeline

Run from the repo root:

```bash
python data_pipeline/scrape_books.py
```

This script:

- scrapes books from `books.toscrape.com`
- cleans and converts GBP to INR using the required fixed baseline rate: 1 GBP = 105.50 INR
- stores the normalized SQLite database at `data_pipeline/books.db`
- prints and saves SQL query outputs under `data_pipeline/query_outputs.json`

## Module 2: Analytics Pipeline

Run from the repo root:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

This pipeline:

- loads the Titanic dataset once from Seaborn's cache/network source
- saves the offline fallback `analytics/titanic.csv`
- cleans the data using threshold-based rules
- produces plots and written interpretation artifacts
- trains and compares classifier models
- saves the complete preprocessing+model pipeline to `analytics/best_pipeline.joblib`

## Module 3: Support Assistant

Run from the repo root:

```bash
cd support_assistant
export MOCK_LLM=1
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then call the endpoint:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is Zepto's delivery policy?"}'
```

The mock mode is the graded baseline and requires no paid service or LLM key.

## Design decisions

### Data pipeline design

The pipeline is intentionally deterministic and dependency-light. It scrapes three categories from a public site, stores a normalized two-table schema, and keeps the conversion rule explicit so the baseline currency conversion is transparent and reproducible.

### Analytics design

The analytics workflow uses one committed offline CSV as the single continuation point for every later step. This avoids reloading the external dataset and makes the analysis stable even when internet access is unavailable.

### Support assistant design

The assistant follows a local RAG pattern: ingest policy documents, embed them into ChromaDB, retrieve the most relevant chunks, and route each query through a LangGraph-based intent router. Mock mode ensures the project works offline with zero API requirements.

## Repository notes

- All written interpretation lives inside Markdown in this repository.
- Generated chart images are supporting artifacts, not a substitute for narrative analysis.
- No paid service is required by any module.
- The git history includes a feature branch merge as part of the repository's required workflow.
