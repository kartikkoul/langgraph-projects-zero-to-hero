# CRAG (Corrective RAG) PDF QnA

A production-style **Corrective RAG** agent that answers questions over your own PDFs. Instead of blindly trusting whatever the vector store returns, the agent **grades** the retrieved chunks, **falls back to web search** when they are weak, **refines** the context down to only the relevant sentences, and then generates a grounded answer.

The whole flow is orchestrated as a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` and exposed through a clean FastAPI service.

---

## Why Corrective RAG?

Naive RAG retrieves the top-k chunks and stuffs them into the prompt, even when those chunks are irrelevant. That leads to hallucinations and "I don't know" answers when the document doesn't cover the question.

CRAG adds a self-correction loop:

1. **Grade** each retrieved chunk for relevance (score `0.0` – `1.0`).
2. Classify the retrieval as `CORRECT`, `AMBIGIOUS`, or `INCORRECT`.
3. **Correct** weak retrievals by rewriting the query and searching the web (Tavily).
4. **Refine** the combined context by keeping only the sentences that actually help.
5. **Generate** the final answer from the cleaned-up context.

---

## What you will practice

- The Corrective RAG (CRAG) pattern end to end
- Building a `StateGraph` with conditional routing in LangGraph
- Document ingestion with LangChain (PDF loading, OCR, sanitizing, chunking, embedding)
- Vector storage and retrieval with `pgvector` (Postgres)
- LLM-based document evaluation using structured outputs
- Query rewriting and web search fallback with Tavily
- Context refinement (sentence-level keep/drop filtering)
- Wrapping an agent behind a clean, framework-agnostic API layer
- Tracing and observability with LangSmith

---

## Architecture

The agent graph (`crag_agent/agent/graph.py`):

![CRAG agent graph](crag_agent/agent/graph.png)

```text
START
  → init                 # set up model, Tavily, score thresholds
  → retrieve             # similarity search over pgvector
  → grade_retrieved_docs # score each chunk, decide CORRECT / AMBIGIOUS / INCORRECT
  → (conditional)
        ├── web_search_required ──→ web_query → web_search ─┐
        └── no_web_search_required ─────────────────────────┤
  → refine_context       # keep only relevant sentences ("strips")
  → generate_answer      # grounded final answer
  → END
```

### The grading verdict

`grade_retrieved_docs` scores every chunk and buckets it using two thresholds (`UPPER_TH = 0.7`, `LOWER_TH = 0.3`):

| Verdict      | Condition                                              | Context used for the answer        |
| ------------ | ------------------------------------------------------ | ---------------------------------- |
| `CORRECT`    | At least one chunk scored `>= UPPER_TH`                | Good docs only                     |
| `AMBIGIOUS`  | No chunk `>= UPPER_TH`, but not all were `<= LOWER_TH` | Good docs **+** web results        |
| `INCORRECT`  | No chunk scored `> LOWER_TH`                           | Web results only                   |

The conditional edge (`crag_agent/agent/routers.py`) triggers web search for any verdict other than `CORRECT`.

---

## Tech stack

- **LangGraph** – agent orchestration (`StateGraph`)
- **LangChain** – document loaders, splitters, embeddings, retriever
- **FastAPI** – HTTP API
- **Postgres + pgvector** – vector store (`langchain-postgres`)
- **OpenAI** – `text-embedding-3-small` embeddings + chat model (via OpenRouter during development to save cost)
- **Tavily** – web search fallback
- **RapidOCR / Unstructured** – image/text extraction from PDFs
- **LangSmith** – tracing (optional)
- **uv** – dependency management

---

The `crag_agent/agent_api_layer.py` is the **only** module the FastAPI app depends on. This keeps the agent reusable from a CLI, worker, or tests without dragging in the web framework.

---

## Setup

### 1. Prerequisites

- Python `>= 3.13`
- [uv](https://docs.astral.sh/uv/) installed
- Docker (for the Postgres + pgvector database)
- API keys: OpenAI (or OpenRouter), Tavily, and optionally LangSmith

### 2. Install dependencies

```bash
cd rag_architectures/crag-pdf-qna
uv sync
```

### 3. Configure environment

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=<api-key>
TAVILY_API_KEY=<api-key>
DB_URI=postgresql+psycopg://postgres:postgres@localhost:5433/crag

# LangSmith (optional tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<api-key>
LANGSMITH_PROJECT="crag-pdf-qna"

# OpenRouter (used by the agent during development to save cost)
OPENROUTER_API_KEY=<api-key>
```

> Note: `init` in `nodes.py` currently uses OpenRouter (`openai/gpt-oss-120b`) to save cost during development. Swap to the commented-out `ChatOpenAI(model="gpt-4.1")` for production.

### 4. Start the vector database

```bash
docker compose up -d
```

This launches `pgvector/pgvector:pg17` with database `crag` exposed on host port `5433`.

### 5. Run the API

```bash
uv run main.py
```

The service starts on `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

---

## Usage

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Ingest PDFs

`POST /ingest` — multipart upload of one or more PDFs. Each file is loaded (with OCR for images), sanitized, chunked (1000 chars, 200 overlap), embedded, and stored in pgvector.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@/path/to/document.pdf"
```

```json
{
  "files": [{ "filename": "document.pdf", "chunks": 42, "ids": ["..."] }],
  "failed": [],
  "total_chunks": 42
}
```

Bad files in a batch are reported under `failed` without aborting the rest.

### Ask a question

`POST /query` — runs the CRAG agent over the ingested documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What game is aezakmi used for?"}'
```

```json
{ "answer": "..." }
```

---

## How the agent works (node by node)

`crag_agent/agent/nodes.py`:

- **`init`** — instantiates the chat model and Tavily client, and seeds the score thresholds (`UPPER_TH`, `LOWER_TH`) into state.
- **`retrieve`** — async similarity search (`k=5`) over the pgvector `documents` collection.
- **`grade_retrieved_docs`** — grades all chunks **concurrently** with `asyncio.gather`, using a structured-output evaluator (`DocEvalScore`). Buckets chunks into good / ambiguous / incorrect and emits a `verdict`.
- **`web_query`** — rewrites the user query into a web-search-friendly query (only reached when correction is needed).
- **`web_search`** — runs Tavily and stores the results.
- **`refine_context`** — picks the context source based on the verdict, splits it into sentences ("strips"), and asks the model to keep or drop each strip (`KeepOrDropStrip`), again concurrently. Only kept strips form the final context.
- **`generate_answer`** — generates the grounded answer from the refined context.

Shared state lives in `GlobalState` (`crag_agent/agent/state_schemas.py`).

---

## Run the agent directly (without the API)

`graph.py` has a `__main__` block useful for local debugging and for regenerating the graph image:

```bash
uv run python -m crag_agent.agent.graph
```

This also saves an updated `graph.png` via `save_graph_visualization`.

---

## Ideas for next additions

- Per-collection/namespace ingestion so multiple document sets can coexist
- Return source citations (chunk metadata + page numbers) alongside the answer
- Streaming responses from `/query`

---