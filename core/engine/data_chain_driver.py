import asyncio

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

from core.engine.data_pipelines.pdf_pipelines import pdf_loader_factory
from core.engine.driver import ChainLLMModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class DataChainDriver:
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(self, model: str = "mistral-api") -> None:
        self.model = ChainLLMModel(model=model)

    def set_data_chains(self, data_chains: list):
        data_chains_list = list()
        for chain in data_chains:
            data_chains_list.append(chain(model=self.model))
        self.data_chains = data_chains_list

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def get_augmented_prompt(
        self,
        query,
        streaming=True,
    ):
        enriched_prompt = """\n"""
        for chain in self.data_chains:
            enriched_prompt += chain.get_data(query)
            enriched_prompt += "\n"
        return enriched_prompt.replace("{", "{{").replace("}", "}}")

    async def aget_augmented_prompt(
        self,
        query,
        streaming=True,
    ):
        tasks = [await chain.aget_data(query) for chain in self.data_chains]
        results = await asyncio.gather(*tasks)
        # enriched_prompt = """\n"""
        # for chain in self.data_chains:
        #    enriched_prompt += await chain.aget_data(query)
        #    enriched_prompt += "\n"

        # results = enriched_prompt
        enriched_prompt = "\n".join(results)
        return enriched_prompt.replace("{", "{{").replace("}", "}}")
