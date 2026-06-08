from typing import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class GlobalState(TypedDict):
    model: ChatOpenAI
    tavily: TavilySearch

    user_query: str

    UPPER_TH: Annotated[float, "Upper threshhold for the relevance score from 0 to 1"]
    LOWER_TH: Annotated[float, "Lower threshhold for the relevance score from 0 to 1"]

    docs: list[Document]
    good_docs: list[Document]

    verdict: Literal["CORRECT", "INCORRECT", "AMBIGIOUS"]
    reason: str

    strips: list[str]
    kept_strips: list[str]
    refined_context: list[str]

    web_query: str
    web_docs: list[Document]

    answer: str