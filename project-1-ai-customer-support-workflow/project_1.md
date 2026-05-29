## Project #1 - AI Customer Support Workflow

### 1) Short Description
This project builds an AI customer support assistant using LangGraph.  
A user asks a support question, the workflow analyzes intent, sentiment, and urgency in parallel, routes the request to the right specialist agent (billing/technical/general), checks response quality, and iterates until the answer is good enough before returning a final response.

Workflow diagram: [`graph.png`](graph.png)

### 2) Assignment (Build It From Scratch)
If you are creating this project from scratch, your assignment is:

1. Define a state schema (`TypedDict`) for all workflow data:
   - user query
   - messages
   - intent, sentiment, urgency
   - draft response, final response
   - retry count and quality score

2. Create prompt classes and output parsers:
   - Pydantic parsers for structured outputs (intent/urgency/quality)
   - String parsers for agent responses

3. Implement graph nodes:
   - model initialization
   - intent/sentiment/urgency analyzers
   - agent assignment message
   - specialist agents (`billing_agent`, `technical_agent`, `general_agent`)
   - quality checker
   - final response node

4. Build a `StateGraph`:
   - Start from `START`
   - Run analyzer nodes in parallel
   - Route conditionally to a specialist agent
   - Add a quality-check loop (retry until score threshold)
   - End at `END`

5. Compile and run the graph

> Note: You can choose to implement or ignore `streamlit_app.py` and `api.py`.  
> They are only for testing/visualizing streaming responses.  
> The main focus of this project is creating the LangGraph agent and executing it.

6. Validate with multiple test queries:
   - billing issue
   - technical issue
   - general query
   - low-quality response requiring retries

### 3) LangGraph Concepts Implemented in the solution
- **StateGraph architecture** with explicit .
- **Shared typed state** via `CustomerAgentState`.
- **Sequential workflow** (start -> setup -> route -> final).
- **Parallel workflow** (intent, sentiment, urgency analyzed concurrently).
- **Conditional routing** using `add_conditional_edges`.
- **Iterative workflow / loop** using quality-based retry logic.
- **Message accumulation reducer** using `add_messages`.

### 4) LangGraph Concepts Not Implemented Yet in the solution (Good Next Additions)
- **Persistence**
- **HITL (Human-In-The-Loop)**
- **Long-term Memory**
- **Fault tolerance and recovery**
- **Observability and tracing**
  - LangSmith tracing/metrics dashboards are integrated on Langsmith branch of this repo, if you want to look switch to that branch.