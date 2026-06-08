import re
from langgraph.graph.state import CompiledStateGraph


def save_graph_visualization(agent: CompiledStateGraph, path: str) -> None:
    png = agent.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png)
    print("Graph visualization saved to graph.png")


def decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if len(s.strip()) > 20]


def doc_text(doc) -> str:
    if isinstance(doc, dict):
        return doc.get("content")
    return doc.page_content
