from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from index import stream_workflow_sse

app = FastAPI(title="Customer Support Workflow API")


class ChatRequest(BaseModel):
    user_query: str


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_workflow_sse(request.user_query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/stream")
def chat_stream_get(user_query: str) -> StreamingResponse:
    return StreamingResponse(
        stream_workflow_sse(user_query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
