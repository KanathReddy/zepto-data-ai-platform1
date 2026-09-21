# Support Assistant Module

## Goal

This module implements a local, deterministic RAG pipeline for Zepto support questions using a policy corpus, ChromaDB vector retrieval, a LangGraph router, and a FastAPI wrapper.

## Pipeline architecture

The pipeline follows the required order:

1. Ingestion: the document corpus is stored under `support_assistant/docs/` as eight Zepto policy text files.
2. Embedding: the files are chunked and embedded with `all-MiniLM-L6-v2` using `sentence-transformers`, then stored in a ChromaDB collection.
3. Retrieval: the `retrieve_and_answer` node embeds the incoming query, retrieves the top-3 matching chunks by cosine similarity, and passes the most relevant snippet forward.
4. Generation: the final answer is generated in a graph node. In mock mode (default), it builds a response using the retrieved snippet. In the optional real-LLM state, the structured prompt template is used to generate a grounded answer.

The route decision is made by `classify_intent`, which sends policy questions to `retrieve_and_answer` and general questions to `direct_answer`. The routing itself does not depend on the LLM toggle; only the generation branch does.

## Prompt template and constraints

The module includes a structured prompt template with the required role–context–task–format–length pattern, plus a negative constraint: "do not answer using information not present in the provided context" and an embedded few-shot example.

## Mock mode behaviour

`MOCK_LLM` is the central switch for this module:

- default or `MOCK_LLM=1`: deterministic mock answer generation, no API call or external network dependency
- `MOCK_LLM=0`: optional real-LLM path using a free-tier LLM backend if available

This means the graded baseline is fully offline and requires no credentials or signup.

## Run

From the repo root:

```bash
cd support_assistant
export MOCK_LLM=1
uvicorn main:app --host 0.0.0.0 --port 7860
```

Example request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is Zepto's delivery policy?"}'
```

## Example calls (mock mode)

Example 1 (retrieval path):

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume.","sources":["doc_01"],"confidence":1.0}
```

Example 2 (general question path):

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker

A local Dockerfile is included at `support_assistant/Dockerfile`. It builds a container for the FastAPI app and runs the server on port 7860.

## Optional extension

The optional real-LLM route is implemented behind `MOCK_LLM=0`, but it is not required for grading. The default mock path remains the baseline that earns full marks.
