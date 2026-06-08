import asyncio
import sys
from pprint import pprint
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from dotenv import load_dotenv

from crag_agent.vectorstore.pgvector_store import get_store

load_dotenv()

# Load Document
@traceable(name="Load document")
async def load_document(file:bytes=None, file_path:str=None, filename:str=None):
    try:
        if file is not None:
            parser = PyPDFParser(
                extract_images=True,
                mode="page",
                images_parser=RapidOCRBlobParser(),
            )

            blob = Blob.from_data(
                file,
                path=filename,
                mime_type="application/pdf",
            )

            return await asyncio.to_thread(lambda: list(parser.lazy_parse(blob)))

        if file_path:
            loader = PyPDFLoader(
                file_path=file_path,
                extract_images=True,
                mode="page",
                images_parser=RapidOCRBlobParser(),
            )
        else:
            loader = PyPDFDirectoryLoader(
                path="docs",
                glob="**/*.pdf",
                extract_images=True,
                mode="page",
                images_parser=RapidOCRBlobParser()
            )

        return await loader.aload()

    except Exception as e:
        pprint(f"Error while loading documents - {e}")
        raise e

#Docs sanitization
@traceable(name="Sanitize docs")
def sanitize_docs(docs: list[Document]) -> None:

    for doc in docs:
        text = doc.page_content

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Modify page's content with the santized text
        doc.page_content = text





#Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


#Embeddings
dense_embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


#Store vectors
@traceable(name="Contextualize & store chunks in Vector DB")
async def store_chunks(chunks: list[Document], collection_name: str):
    """ Fetches the store -> Automatically generates embeddings for each chunk -> stores in the vector DB  """
    store = get_store(dense_embeddings_model, collection_name)

    ids = []
    for index, chunk in enumerate(chunks):
        # PyPDFLoader doesn't reliably set "title"; fall back to the source filename.
        raw_title = chunk.metadata.get("title") or chunk.metadata.get("source") or "doc"
        title = re.sub(r"[^a-zA-Z0-9]+", "_", str(raw_title)).strip("_").lower() or "doc"
        ids.append(f"{title}_{index}")

    result = await store.aadd_documents(
        documents=chunks,
        ids=ids
    )

    return result

#Ingestion Pipeline
@traceable(name="Ingestion pipeline")
async def process_docs(file:bytes=None, file_path:str=None, filename:str=None, collection_name:str="documents"):

    docs = await load_document(file=file, file_path=file_path, filename=filename)

    sanitize_docs(docs)

    chunks = splitter.split_documents(docs) 

    ids = await store_chunks(chunks, collection_name)
    
    return ids
    

# psycopg async mode is incompatible with Windows' default ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    asyncio.run(process_docs())