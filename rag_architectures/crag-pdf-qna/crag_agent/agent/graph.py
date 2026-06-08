import asyncio
from pprint import pprint
from langgraph.graph import END, START, StateGraph
from crag_agent.agent.routers import is_websearch_required
from crag_agent.agent.state_schemas import GlobalState
from crag_agent.agent.nodes import grade_retrieved_docs, init, retrieve, web_query, web_search, refine_context, generate_answer
from crag_agent.agent.utils import save_graph_visualization

crag_agent_builder = StateGraph(GlobalState)

# Nodes
crag_agent_builder.add_node("init", init)
crag_agent_builder.add_node("retrieve", retrieve)
crag_agent_builder.add_node("grade_retrieved_docs", grade_retrieved_docs)
crag_agent_builder.add_node("web_query", web_query)
crag_agent_builder.add_node("web_search", web_search)
crag_agent_builder.add_node("refine_context", refine_context)
crag_agent_builder.add_node("generate_answer", generate_answer)

# Edges
crag_agent_builder.add_edge(START, "init")
crag_agent_builder.add_edge("init", "retrieve")
crag_agent_builder.add_edge("retrieve", "grade_retrieved_docs")

crag_agent_builder.add_conditional_edges("grade_retrieved_docs", is_websearch_required, {
    "web_search_required": "web_query",
    "no_web_search_required": "refine_context"
})

crag_agent_builder.add_edge("web_query", "web_search")
crag_agent_builder.add_edge("web_search", "refine_context")

crag_agent_builder.add_edge("refine_context", "generate_answer")
crag_agent_builder.add_edge("generate_answer", END)

#Compile
crag_agent = crag_agent_builder.compile()

if __name__ == "__main__":

    # Uncomment the below code when saving graph visualizetion is required
    from pathlib import Path
    current_dir = Path(__file__).parent
    save_graph_visualization(crag_agent, current_dir / "graph.png")
    # ====================================================================

    result = asyncio.run(crag_agent.ainvoke({
        "user_query": "What game is aezakmi used for?"
    }))

    pprint(result["answer"])