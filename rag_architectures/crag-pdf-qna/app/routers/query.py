from fastapi import APIRouter

from app.models.query import QueryRequest, QueryResponse
from app.services.query import query_crag


router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = await query_crag(request.query)
    return QueryResponse(answer=result)