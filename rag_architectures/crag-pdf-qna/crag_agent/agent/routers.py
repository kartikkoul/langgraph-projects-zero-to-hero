from crag_agent.agent.state_schemas import GlobalState


def is_websearch_required(state: GlobalState) -> bool:
    """ Check if web search is required """
    return "web_search_required" if state["verdict"] != "CORRECT" else "no_web_search_required"