from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from pydantic import Field


class CustomerAgentState(TypedDict):
    user_query: str
    model: ChatOpenAI
    messages: Annotated[list, add_messages]
    intent: str
    sentiment: str
    urgency: int
    draft_response: str
    final_response: str
    final_response_quality_score: int
    retry_count: int