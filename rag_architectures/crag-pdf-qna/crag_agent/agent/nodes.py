""" Nodes of the crag_agent """
import asyncio
import os

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from crag_agent.agent.prompts import AugmentedPrompt, DocsEvalPrompt, RefineContextPrompt, WebQueryPrompt
from crag_agent.agent.schemas import DocEvalScore, KeepOrDropStrip
from crag_agent.agent.state_schemas import GlobalState
from crag_agent.agent.utils import decompose_to_sentences, doc_text
from crag_agent.retrieval.index import retrieve_docs

def init(state: GlobalState) -> GlobalState:
    # model= ChatOpenAI(
    #     model= "gpt-4.1",
    #     temperature= 0  
    # )

    # Development (to save costs)
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key = os.environ["OPENROUTER_API_KEY"],
        temperature = 0,
        model="openai/gpt-oss-120b"
    )

    tavily = TavilySearch(
        max_results = 5
    )

    return {
        "model": model,
        "tavily": tavily,
        "UPPER_TH": 0.7,
        "LOWER_TH": 0.3,
    }

async def retrieve(state: GlobalState) -> GlobalState:
    docs = await retrieve_docs(state["user_query"])

    return {
        "docs": docs
    }

async def grade_retrieved_docs(state: GlobalState) -> GlobalState:
    model = state["model"]
    prompt = DocsEvalPrompt()
    evaluator = prompt.prompt_template | model.with_structured_output(DocEvalScore)

    user_query = state["user_query"]
    docs = state["docs"]


    evals: list[DocEvalScore] = await asyncio.gather(
        *[
            evaluator.ainvoke(
                {
                    "user_query": user_query,
                    "chunk_page_content": doc.page_content,
                }
            )
            for doc in docs
        ]
    )

    upper_th = state["UPPER_TH"]
    lower_th = state["LOWER_TH"]

    good_docs = []
    ambigious_docs = []
    incorrect_docs = []

    for doc, eval in zip(docs, evals):
        score = eval.score
        if score >= upper_th:
            good_docs.append(doc)
        elif score <= lower_th:
            incorrect_docs.append(doc)
        else:
            ambigious_docs.append(doc)



    if len(good_docs) > 0:
        return {
            "good_docs": good_docs,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {upper_th}"
        }

    elif len(ambigious_docs) == 0:
        return {
            "good_docs": good_docs,
            "verdict": "INCORRECT",
            "reason": f"No retrieved chunk > {lower_th}"
        }
    else:
        return {
            "good_docs": good_docs,
            "verdict": "AMBIGIOUS",
            "reason": f"No retrieved chunk > {upper_th} but not all were < {lower_th}"
        }

async def web_query(state: GlobalState) -> GlobalState:
    """ Rewrite query for web search """
    model = state["model"]
    prompt = WebQueryPrompt()
    web_query_generator = prompt.prompt_template | model | prompt.parser

    user_query = state["user_query"]
    web_query = await web_query_generator.ainvoke(
        {
            "user_query": user_query
        }
    )
    return {
        "web_query": web_query
    }

async def web_search(state: GlobalState) -> GlobalState:
    """ Search the web for the web query """
    tavily = state["tavily"]
    web_query = state["web_query"]
    result = await tavily.ainvoke(web_query)

    return {
        "web_docs": result.get("results", [])
    }

async def refine_context(state: GlobalState) -> GlobalState:
    model = state["model"]
    prompt = RefineContextPrompt()
    strip_decider = prompt.prompt_template | model.with_structured_output(KeepOrDropStrip)

    docs_to_use = []
    if state["verdict"] == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state["verdict"] == "AMBIGIOUS":
        docs_to_use = state["good_docs"] + state["web_docs"]
    else:
        docs_to_use = state["web_docs"]

    

    context = "\n\n".join([doc_text(doc) for doc in docs_to_use])
    user_query = state["user_query"]

    strips = decompose_to_sentences(context)
    
    async def decide_strip(strip: str):
        return await strip_decider.ainvoke(
            {
                "user_query": user_query,
                "strip": strip
            }
        )

    strip_decisions = await asyncio.gather(*[decide_strip(s) for s in strips])

    kept_strips = [strip for strip, decision in zip(strips, strip_decisions) if decision.keep]

    refined_context = "\n\n".join(kept_strips).strip()

    return {
        "strips": strips,
        "kept_strips": kept_strips,
        "refined_context": refined_context
    }

async def generate_answer(state: GlobalState) -> GlobalState:
    model = state["model"]
    prompt = AugmentedPrompt()
    answer_generator = prompt.prompt_template | model | prompt.parser

    user_query = state["user_query"]
    context = state["refined_context"]
    answer = await answer_generator.ainvoke(
        {
            "user_query": user_query,
            "context": context
        }
    )
    
    return {
        "answer": answer
    }