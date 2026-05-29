from state_schemas import CustomerAgentState


def agent_assigner_router(state: CustomerAgentState):
    return state["intent"]


def quality_response_router(state: CustomerAgentState):
    if state["final_response_quality_score"] < 7 and state["retry_count"] <= 5:
        return state["intent"]
    else:
        return "continue"