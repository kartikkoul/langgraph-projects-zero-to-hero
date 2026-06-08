from abc import ABC
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from crag_agent.agent.schemas import DocEvalScore, KeepOrDropStrip


class BasePromptClass(ABC):
    parser: PydanticOutputParser | StrOutputParser
    prompt_template: ChatPromptTemplate


class DocsEvalPrompt(BasePromptClass):
    parser = PydanticOutputParser(pydantic_object=DocEvalScore)
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
            You are a strict retrieval evaluator for RAG.
            You will be given one retrieved chunk & a question and you have to score relevance of the chunk to the question.

            Rules:
            Be deterministic about your judgement & scores.

            \n{format_instructions}
        """,
            ),
            ("human", "Questions: {user_query}\nChunk: {chunk_page_content}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

class WebQueryPrompt(BasePromptClass):
    parser = StrOutputParser()
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                Rewrite the question into a web search query composed of keywords.
                \nRules:
                \n- Keep it short (6-20 words)
                \n- If the question implies recency (e.g, recent/latest/last week/last month), add a constraint like (last 30 days)
                \n- DO NOT ANSWER the question.
                \n- Return a single string query.
                """
            ),
            ("human", "Question: {user_query}"),
        ]
    )


class RefineContextPrompt(BasePromptClass):
    parser = PydanticOutputParser(pydantic_object=KeepOrDropStrip)
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """ 
                    You are a strict strip evaluator for RAG. 
                    You will be given a strip of text and a question and you have to decide whether to keep the strip or drop it and along with the reason for the decision.
                    \n{format_instructions}
                 """
            ),
            ("human", "Question: {user_query}\nChunk: {strip}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())


class AugmentedPrompt(BasePromptClass):
    parser = StrOutputParser()
    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                You are a helpful assistant that answers questions based on the context provided.
                If the context is not enough to answer the question, simply say, "I don't know" or "I don't have the information to answer that question".

                \nRULES:
                \n- Be deterministic about your answers.
                \n- Be concise and to the point.
                \n- Do not make up information.
                \n- Do not provide any information that is not in the context.
                \n- Keep your answer grounded in the context provided.
                """
            ),
            ("human", "Question: {user_query}\nContext: {context}"),
        ]
    )