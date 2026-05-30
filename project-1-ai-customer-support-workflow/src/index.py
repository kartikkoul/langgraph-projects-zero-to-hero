import json
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    analyze_intent,
    analyze_sentiment,
    analyze_urgency,
    assign_respective_agent,
    billing_agent,
    check_response_quality,
    final_response,
    general_agent,
    initiate_model,
    technical_agent,
)
from src.routing_functions import agent_assigner_router, quality_response_router
from src.state_schemas import CustomerAgentState

load_dotenv()

# Graph initiation
main_agent_builder = StateGraph(CustomerAgentState)


# Graph formation
# Nodes
main_agent_builder.add_node("initiate_model", initiate_model)

main_agent_builder.add_node("analyze_intent", analyze_intent)
main_agent_builder.add_node("analyze_sentiment", analyze_sentiment)
main_agent_builder.add_node("analyze_urgency", analyze_urgency)

main_agent_builder.add_node("assign_respective_agent", assign_respective_agent)

main_agent_builder.add_node("technical_agent", technical_agent)
main_agent_builder.add_node("billing_agent", billing_agent)
main_agent_builder.add_node("general_agent", general_agent)

main_agent_builder.add_node("check_response_quality", check_response_quality)

main_agent_builder.add_node("final_response", final_response)

# Edges
main_agent_builder.add_edge(START, "initiate_model")

main_agent_builder.add_edge("initiate_model", "analyze_intent")
main_agent_builder.add_edge("initiate_model", "analyze_sentiment")
main_agent_builder.add_edge("initiate_model", "analyze_urgency")

main_agent_builder.add_edge("analyze_intent", "assign_respective_agent")
main_agent_builder.add_edge("analyze_sentiment", "assign_respective_agent")
main_agent_builder.add_edge("analyze_urgency", "assign_respective_agent")

main_agent_builder.add_conditional_edges(
    "assign_respective_agent",
    agent_assigner_router,
    {
        "technical_agent": "technical_agent",
        "billing_agent": "billing_agent",
        "general_agent": "general_agent",
    },
)


main_agent_builder.add_edge("technical_agent", "check_response_quality")
main_agent_builder.add_edge("billing_agent", "check_response_quality")
main_agent_builder.add_edge("general_agent", "check_response_quality")

main_agent_builder.add_conditional_edges(
    "check_response_quality",
    quality_response_router,
    {
        "technical_agent": "technical_agent",
        "billing_agent": "billing_agent",
        "general_agent": "general_agent",
        "continue": "final_response",
    },
)

main_agent_builder.add_edge("final_response", END)


# Graph compilation
main_agent = main_agent_builder.compile()


# SSE Logic
def _build_node_payload(node_state: Optional[CustomerAgentState]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if not node_state:
        return payload

    if "intent" in node_state:
        payload["intent"] = "Analyzing intent..."

    if "sentiment" in node_state:
        payload["sentiment"] = "Analyzing sentiment..."

    if "urgency" in node_state:
        payload["urgency"] = "Analyzing urgency..."

    if "final_response_quality_score" in node_state:
        payload["quality_score"] = "Analyzing my answer..."

    if "messages" in node_state and node_state["messages"]:
        last_message = node_state["messages"][-1]
        last_message_type = str(getattr(last_message, "type", "")).lower()
        if last_message_type == "ai":
            payload["routed_message"] = getattr(last_message, "content", str(last_message))

    if "draft_response" in node_state:
        payload["draft_response"] = str(node_state["draft_response"])

    if "final_response" in node_state:
        payload["final_response"] = str(node_state["final_response"])

    return payload


def stream_workflow_events(
    user_query: str
) -> Iterator[Dict[str, Any]]:
    """Yield workflow updates in a frontend-friendly event structure."""
    final_answer = None

    for update in main_agent.stream(
        {"user_query": user_query},
        stream_mode="updates",
    ):
        for node_name, node_state in update.items():
            payload = _build_node_payload(node_state)
            yield {"type": "node_update", "node": node_name, "payload": payload}

            if node_state and "final_response" in node_state:
                final_answer = node_state["final_response"]

    if final_answer:
        yield {"type": "final_response", "payload": {"text": str(final_answer)}}
    else:
        yield {
            "type": "final_response",
            "payload": {"text": "Could not generate a final response."},
        }


def stream_workflow_sse(user_query: str) -> Iterator[str]:
    """Yield Server-Sent Events (SSE) for frontend EventSource consumers."""
    for event in stream_workflow_events(user_query):
        event_type = event["type"]
        event_payload = {key: value for key, value in event.items() if key != "type"}
        yield f"event: {event_type}\ndata: {json.dumps(event_payload, default=str)}\n\n"

    yield "event: done\ndata: {}\n\n"


def _render_node_update(node_name: str, payload: Dict[str, Any]) -> None:
    print(f"\n[Node] {node_name}")

    if "quality_score" in payload:
        print(f"  quality_score: {payload['quality_score']}")

    if "routed_message" in payload:
        print(f"  routed_message: {payload['routed_message']}")

    if "draft_response" in payload:
        print(f"  draft_response: {payload['draft_response']}")


def save_graph_visualization(output_path: str = "graph.png") -> None:
    png_bytes = main_agent.get_graph(xray=True).draw_mermaid_png()
    with open(output_path, "wb") as graph_file:
        graph_file.write(png_bytes)
        print("Check the graph flow at graph.png\n")
        graph_file.close()


def run_cli_chat() -> None:
    print("Enter your query (type 'END' to end the chat):: ")
    while True:
        user_input = input("\nYou:: ")

        if user_input == "END":
            break

        print("Bot:: Thinking...")
        final_answer = None

        for event in stream_workflow_events(user_input):
            if event["type"] == "node_update":
                _render_node_update(event["node"], event["payload"])
            elif event["type"] == "final_response":
                final_answer = event["payload"]["text"]

        print(f"\nBot:: {final_answer}")

  

if __name__ == "__main__":
    # Uncomment if you want to download graph diagram
    # save_graph_visualization() 

    run_cli_chat()
