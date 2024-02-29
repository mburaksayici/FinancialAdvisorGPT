"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os
import requests
import logging

from database.vector_db import chromadb_langchain_client


class DBRetrievalPipeline:
    def __init__(self):
        self.chromadb_langchain_client = chromadb_langchain_client

    def query(self, query, k):
        docs = self.chromadb_langchain_client.similarity_search(query, k=k)
        logging.info(f"Retrieved {len(docs)} documents from the database")
        return [
            {"page_content": doc.page_content, "source": doc.metadata["source"]}
            for doc in docs
        ]
