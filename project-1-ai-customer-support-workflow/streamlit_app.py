import json
from typing import Any, Dict, Iterator, List

import requests
import streamlit as st


def iter_sse_events(response: requests.Response) -> Iterator[Dict[str, Any]]:
    """Parse an SSE response into structured event dictionaries."""
    event_name = "message"
    data_lines: List[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        line = raw_line.strip()

        if not line:
            if data_lines:
                raw_payload = "\n".join(data_lines)
                try:
                    payload: Any = json.loads(raw_payload)
                except json.JSONDecodeError:
                    payload = {"raw": raw_payload}

                yield {"event": event_name, "data": payload}

            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())


def stream_backend_events(base_url: str, user_query: str) -> Iterator[Dict[str, Any]]:
    """Connect to FastAPI SSE endpoint and yield events."""
    stream_url = f"{base_url.rstrip('/')}/chat/stream"

    with requests.get(
        stream_url,
        params={"user_query": user_query},
        stream=True,
        timeout=(10, 600),
    ) as response:
        response.raise_for_status()
        yield from iter_sse_events(response)


def extract_live_update(payload: Dict[str, Any]) -> str:
    for key in ("intent", "sentiment", "urgency", "quality_score"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def normalize_routed_message(raw_message: Any) -> str:
    if raw_message is None:
        return ""

    message = str(raw_message).strip()
    if not message:
        return ""

    if message.lower().startswith("route:"):
        message = message.split(":", 1)[1].strip()

    return message


def run_app() -> None:
    st.set_page_config(page_title="Customer Support Chat", page_icon="💬", layout="wide")
    st.title("💬 Customer Support Workflow")
    st.caption("Test UI to stream response from our AI workflow.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Connection")
        backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your support question...")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        routed_placeholder = st.empty()
        answer_placeholder = st.empty()

        live_update = ""
        routed_message = ""
        final_answer = ""

        def render_live_update() -> None:
            if not live_update:
                status_placeholder.empty()
                return

            status_placeholder.markdown(
                f"<div style='opacity: 0.65;'>{live_update}</div>",
                unsafe_allow_html=True,
            )

        try:
            for event in stream_backend_events(backend_url, prompt):
                event_type = event.get("event")
                event_data = event.get("data", {})

                if event_type == "node_update":
                    payload = event_data.get("payload", {})
                    routed_candidate = normalize_routed_message(
                        payload.get("routed_message")
                    )
                    if routed_candidate:
                        routed_message = routed_candidate
                        routed_placeholder.markdown(routed_message)


                    new_live_update = extract_live_update(payload)
                    if new_live_update:
                        live_update = new_live_update
                        render_live_update()

                elif event_type == "final_response":
                    payload = event_data.get("payload", {})
                    final_answer = payload.get(
                        "text", "Could not generate a final response."
                    )
                    status_placeholder.empty()
                    answer_placeholder.markdown(final_answer)

                elif event_type == "done":
                    break

            status_placeholder.empty()
            if not final_answer:
                final_answer = "Could not generate a final response."
                answer_placeholder.markdown(final_answer)

        except requests.RequestException as request_error:
            status_placeholder.empty()
            final_answer = f"Backend request failed: {request_error}"
            answer_placeholder.error(final_answer)
        except Exception as unknown_error:  # noqa: BLE001
            status_placeholder.empty()
            final_answer = f"Unexpected error: {unknown_error}"
            answer_placeholder.error(final_answer)

    assistant_content = (
        f"{routed_message}\n\n{final_answer}" if routed_message else final_answer
    )
    st.session_state.messages.append({"role": "assistant", "content": assistant_content})


if __name__ == "__main__":
    run_app()
