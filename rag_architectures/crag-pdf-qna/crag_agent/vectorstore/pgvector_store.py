import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

def get_store(embeddings: OpenAIEmbeddings,collection_name:str):
    store = PGVector(
        embeddings= embeddings,
        collection_name= collection_name,
        connection= os.environ["DB_URI"],
        use_jsonb= True,
        async_mode= True
    )

    return store