"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os
import requests
from logging_stack import logger
import asyncio

from database.vector_db import chromadb_langchain_client


class DBRetrievalPipeline:
    def __init__(self):
        self.chromadb_langchain_client = chromadb_langchain_client

    def query(self, query, k):
        docs = self.chromadb_langchain_client.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(docs)} documents from the database")
        return [
            {"page_content": doc.page_content, "source": doc.metadata["source"]}
            for doc in docs
        ]

    async def aquery(self, query, k):
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, self._query_sync, query, k)
        logger.info(f"Retrieved {len(docs)} documents from the database")
        return [
            {
                "page_content": doc.page_content,
                "source": doc.metadata["source"],
                "query": query,
            }
            for doc in docs
        ]

    def _query_sync(self, query, k):
        return self.chromadb_langchain_client.similarity_search(query, k=k)
