from logging_stack import logger
from queue import Empty
from threading import Thread

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

from core.engine.data_pipelines.pdf_pipelines import pdf_loader_factory

# from core.engine.llm_models.mistral_7b_engine import Mistral7BInstructModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class ModelDriver:
    # Prompt
    TEMPLATE = """
    I would like you to create a detailed report. Report should be based on the financial document I'm giving you between two *** . It is at the below.


***

 {context}

***

    
    Here's the additional question: {question}
    Helpful Answer:"""
    # TEMPLATE="Short answers only"
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=TEMPLATE,
    )

    def __init__(self) -> None:
        self.model = None
        self.set_pdf_loader()  # to be fixed later
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

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
            self.model = MistralAPIModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.

    async def chat(
        self,
        query,
        streaming=True,
    ):
        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )
        # if streaming:
        async for chunk in qa_chain.astream(
            {"query": query}
        ):  #         async for chunk in qa_chain.astream({"query": query}):
            yield chunk["result"]
        # else:F
        #    return qa_chain({"query": query})['result']

    def get_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )
        return qa_chain

    async def chat_chain_injected(self, query, qa_chain, streaming=True):

        # if streaming:
        async for chunk in qa_chain.astream(
            {"query": query}
        ):  #         async for chunk in qa_chain.astream({"query": query}):
            yield chunk["result"]
        # else:F
        #    return qa_chain({"query": query})['result']


from langchain.chains import LLMChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# TO DO : Reduce model drivers to one. Or, may be we can have two drivers.
class ChainLLMModel:  # TO DO : Better name
    def __init__(self, model="mistral-api") -> None:
        self.load_model(model)
        self.set_embedding()  # to be fixed later
        self.set_pdf_loader()  # to be fixed later
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def load_model(self, model_name):  # TO DO : Move model DB to mongo.
        if model_name == "mistral-api":
            self.model = MistralAPIModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.

    async def chat(
        self,
        query,
        streaming=True,
    ):
        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )
        # if streaming:
        async for chunk in qa_chain.astream({"query": query}):
            yield chunk["result"]

    def nonasync_chat(self, query, prompt_template):

        chain = LLMChain(llm=self.model, prompt=prompt_template)

        return chain.run(query)  # qa_chain({"query": query})['result']

    async def async_chat(self, query, prompt_template):

        chain = LLMChain(llm=self.model, prompt=prompt_template)

        return await chain.arun(query)  # qa_chain({"query": query})['result']
