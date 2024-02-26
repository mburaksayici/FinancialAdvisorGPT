import logging
from queue import Empty
from threading import Thread

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

from core.engine.data_chain_driver import DataChainDriver
from core.engine.data_pipelines.pdf_pipelines import pdf_loader_factory

# from core.engine.llm_models.mistral_7b_engine import Mistral7BInstructModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class OnlineModelDriver:
    def get_template(
        self,
    ):
        return """
    I would like you to create a detailed report. Please use and cite the sources with the links/infos/urls/authors I've shared, Report should be based on the financial document I'm giving you between two *** . It is at the below.


***

 {context}

***

    
    Here's the additional question: {question}

    """

    def get_qa_chain_prompt(self, template):
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def __init__(self) -> None:
        self.model = None
        self.data_chains = None
        self.set_pdf_loader()  # to be fixed later
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later
        self.data_chain_driver = DataChainDriver()

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def set_data_chains(self, data_chains: list):
        self.data_chain_driver.set_data_chains(data_chains)

    def load_document(self, document_link, is_bytes=False):
        # document_link = "https://www.apple.com/newsroom/pdfs/FY23_Q1_Consolidated_Financial_Statements.pdf"
        if not is_bytes:
            loader = self.loader("uploaded_files/" + document_link)
        else:
            loader = self.loader(document_link)
        data = loader.load()
        all_splits = self.text_splitter.split_documents(data)
        self.vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=self.embedding
        )

    def load_model(self, model_name):  # TO DO : Move model DB to mongo.
        # if model_name == "mistral":
        #    self.model = Mistral7BInstructModel(
        #        ""
        #    ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.
        if model_name == "mistral-api":
            print("using api model")
            self.model = MistralAPIModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.

    def chat(
        self,
        query,
        streaming=True,
    ):

        if self.data_chain_driver.data_chains:  # Fix later.
            enriched_prompt = self.data_chain_driver.get_augmented_prompt(query)

        template = self.get_template()
        template += enriched_prompt

        qa_chain_prompt = self.get_qa_chain_prompt(template)

        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

        # if streaming:
        # async for chunk in qa_chain.astream(
        #    {"query": query}
        # ):  #         async for chunk in qa_chain.astream({"query": query}):
        #    yield chunk["result"]
        # else:F
        return qa_chain({"query": query})["result"]
