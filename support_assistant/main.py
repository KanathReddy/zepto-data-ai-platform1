from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(__file__).resolve().parent / "docs"
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1" or os.getenv("MOCK_LLM") is None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class GraphState(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    chunks: List[str]
    sources: List[str]
    answer: str
    confidence: float


def load_documents() -> dict[str, str]:
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def build_chunks(docs: dict[str, str]) -> dict[str, str]:
    chunks = {}
    for doc_id, text in docs.items():
        chunk = text.strip()
        chunks[doc_id] = chunk
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="zepto_policy_collection")
    return collection


def ensure_indexed():
    docs = load_documents()
    chunks = build_chunks(docs)
    collection = get_collection()
    ids = list(chunks.keys())
    existing = collection.get(ids=ids, include=[])
    if existing["ids"]:
        return collection
    collection.add(
        ids=ids,
        embeddings=EMBED_MODEL.encode(list(chunks.values())).tolist(),
        documents=list(chunks.values()),
    )
    return collection


def classify_intent(query: str) -> str:
    q = query.lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    if any(keyword in q for keyword in keywords):
        return "policy_question"
    return "general_question"


def retrieve_chunks(query: str, top_k: int = 3):
    collection = ensure_indexed()
    embeds = EMBED_MODEL.encode([query]).tolist()
    result = collection.query(query_embeddings=embeds, n_results=top_k, include=["documents", "distances"])
    docs = result["documents"][0]
    ids = result["ids"][0]
    return list(zip(ids, docs))


def build_mock_answer(query: str, matched: list[tuple[str, str]]) -> str:
    if not matched:
        return "I can only answer questions about Zepto policies right now."
    top_id, top_chunk = matched[0]
    snippet = top_chunk[:200]
    return f"Based on the retrieved context: {snippet}"


def generate_policy_answer(query: str, matched: list[tuple[str, str]]) -> QueryResponse:
    if MOCK_LLM:
        answer = build_mock_answer(query, matched)
        ids = [doc_id for doc_id, _ in matched]
        return QueryResponse(answer=answer, sources=ids, confidence=1.0)

    top_id, top_chunk = matched[0] if matched else ("", "")
    prompt = f"""
    Role: You are a Zepto policy assistant.
    Context: {top_chunk}
    Task: Answer the user's question based only on the provided context.
    Format: Return a concise, grounded answer.
    Length: 2-5 sentences maximum.
    Negative constraint: Do not answer using information not present in the provided context.
    Example: User: What is the delivery fee? Assistant: Standard delivery is free on orders over INR 149; orders below that threshold incur a flat INR 25 fee.
    User question: {query}
    """
    # Optional real-LLM extension is intentionally left as a deterministic placeholder for compatibility.
    answer = f"Based on the retrieved context: {top_chunk[:200]}"
    return QueryResponse(answer=answer, sources=[top_id] if top_id else [], confidence=1.0)


def generate_direct_answer(query: str) -> QueryResponse:
    if MOCK_LLM:
        return QueryResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)
    return QueryResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)


def classify_node(state: GraphState) -> GraphState:
    state["intent"] = classify_intent(state["query"])
    return state


def retrieve_node(state: GraphState) -> GraphState:
    matches = retrieve_chunks(state["query"])
    state["chunks"] = [chunk for _, chunk in matches]
    state["sources"] = [doc_id for doc_id, _ in matches]
    response = generate_policy_answer(state["query"], matches)
    state["answer"] = response.answer
    state["confidence"] = response.confidence
    return state


def direct_node(state: GraphState) -> GraphState:
    response = generate_direct_answer(state["query"])
    state["answer"] = response.answer
    state["confidence"] = response.confidence
    state["sources"] = []
    return state


def route_after_classification(state: GraphState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_node)
workflow.add_node("retrieve_and_answer", retrieve_node)
workflow.add_node("direct_answer", direct_node)
workflow.add_conditional_edges("classify_intent", route_after_classification, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer",
})
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
workflow.set_entry_point("classify_intent")
app_graph = workflow.compile()

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    state = {"query": request.query, "intent": "policy_question", "chunks": [], "sources": [], "answer": "", "confidence": 1.0}
    result = app_graph.invoke(state)
    return QueryResponse(answer=result["answer"], sources=result.get("sources", []), confidence=float(result.get("confidence", 1.0)))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
