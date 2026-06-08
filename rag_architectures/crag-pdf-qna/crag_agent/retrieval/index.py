from crag_agent.vectorstore.pgvector_store import get_store
from crag_agent.ingestion.index import dense_embeddings_model

def get_retriever(collection_name: str):
    store = get_store(
        embeddings=dense_embeddings_model,
        collection_name=collection_name
    )

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

retriever = get_retriever("documents")

async def retrieve_docs(query: str):
    docs = await retriever.ainvoke(query)
    return docs