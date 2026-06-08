from crag_agent.agent_api_layer import query_agent


async def query_crag(query: str) -> str:
    result = await query_agent(query)
    return result