import os

import chromadb
from langchain_community.vectorstores import Chroma

from core.engine.embeddings.hf_embeddings import embedding

chromadb_path = os.getenv(
    "LOCAL_CHROMADB_PATH"
)  ## NOTE : This is for persistend db storage. Clients with serving to be added

chromadb_path = os.path.join(os.getcwd(), chromadb_path)
chromadb_client = chromadb.PersistentClient(
    path=chromadb_path,
)


chromadb_langchain_client = Chroma(
    embedding_function=embedding,
    client=chromadb_client,
)
