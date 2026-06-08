"""Public surface of the CRAG agent.

The API layer (FastAPI) should depend ONLY on this module, never on the
internals (ingestion pipeline, vectorstore, graph nodes). This keeps the agent
reusable from a CLI, worker, or tests without dragging in any web framework.
"""

import asyncio

from fastapi import HTTPException
from crag_agent.agent.graph import crag_agent
from crag_agent.agent.schemas import StandardError
from crag_agent.ingestion.index import process_docs

async def ingest_pdf_bytes(
    content: bytes,
    filename: str,
) -> list[str]:
    """Run the ingestion pipeline for uploaded doc via API.
    """
    try:
        ids = await process_docs(
            file=content,
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=StandardError(
            message = "Internal server error",
            error = str(e),
            code = 500
        ))

    return ids

async def query_agent(query: str) -> str:
    result = await crag_agent.ainvoke({
        "user_query": query
    })
    
    return result["answer"]