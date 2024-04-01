from logging_stack import logger
from queue import Empty
from threading import Thread
import traceback
import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.engine.message_prompts.message_prompts import chat_prompt
from core.engine.data_chain_driver import DataChainDriver
from core.engine.data_pipelines.pdf_pipelines import pdf_loader_factory
from core.engine.conversation.conversation_manager import ConversationManager
from database.vector_db import chromadb_client
from models import *

# from core.engine.llm_models.mistral_7b_engine import Mistral7BInstructModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class OnlineModelDriver:
    def __init__(self) -> None:
        self.model = None
        self.data_chains = None
        self.chromadb_client = (
            chromadb_client  # vectorstore, it can be injected through fastapi as well.
        )
        self.set_pdf_loader()  # to be fixed later
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later
        self.chat_prompt = chat_prompt
        self.data_chain_driver = DataChainDriver()
        self.conversation_manager = ConversationManager()

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def set_data_chains(self, data_chains: list):
        self.data_chain_driver.set_data_chains(data_chains)

    def initialise_conversation(self, user_id):
        return self.conversation_manager.initialise_conversation(user_id=user_id)

    def load_document(self, document_link, is_bytes=False):
        # document_link = "https://www.apple.com/newsroom/pdfs/FY23_Q1_Consolidated_Financial_Statements.pdf"
        if not is_bytes:
            loader = self.loader("uploaded_files/" + document_link)
        else:
            loader = self.loader(document_link)
        data = loader.load()
        all_splits = self.text_splitter.split_documents(data)
        self.chroma_collection = Chroma.from_documents(
            documents=all_splits,
            embedding=self.embedding,
            client=self.chromadb_client,
        )

    def load_assets(
        self,
    ):
        # TO DO : add path to config or env file.
        uploaded_files_folder = "uploaded_files/"
        logger.info("Assets directory is loading.")
        for pdf in os.listdir(uploaded_files_folder):
            if ".pdf" in pdf:
                try:
                    self.load_document(pdf)
                except Exception as exc:
                    logger.warning(f"Couldnt load {pdf} file :: {exc}", exc_info=True)

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

    def answer(
        self,
        query,
        streaming=True,
    ):
        if self.data_chain_driver.data_chains:  # Fix later.
            enriched_prompt = self.data_chain_driver.get_augmented_prompt(query)
        formatted_chat_prompt = self.chat_prompt.format_messages(
            question=query, context=enriched_prompt
        )

        # if streaming:
        # async for chunk in qa_chain.astream(
        #    {"query": query}
        # ):  #         async for chunk in qa_chain.astream({"query": query}):
        #    yield chunk["result"]
        # else:F
        try:
            result = self.model.invoke(formatted_chat_prompt)
            return result
        except Exception:
            logger.error(traceback.format_exc())

    async def aanswer(
        self,
        query,
        streaming=True,
    ):
        if self.data_chain_driver.data_chains:  # Fix later.
            enriched_prompt = await self.data_chain_driver.aget_augmented_prompt(query)
        formatted_chat_prompt = self.chat_prompt.format_messages(
            question=query, context=enriched_prompt
        )

        # if streaming:
        # async for chunk in qa_chain.astream(
        #    {"query": query}
        # ):  #         async for chunk in qa_chain.astream({"query": query}):
        #    yield chunk["result"]
        # else:F
        try:
            result = self.model.invoke(formatted_chat_prompt)
            return result
        except Exception:
            logger.error(traceback.format_exc())

    def chatt(
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
            retriever=self.chroma_collection.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

        # if streaming:
        # async for chunk in qa_chain.astream(
        #    {"query": query}
        # ):  #         async for chunk in qa_chain.astream({"query": query}):
        #    yield chunk["result"]
        # else:F
        try:
            result = qa_chain({"query": query})["result"]
            return result
        except Exception:
            logger.error(traceback.format_exc())

    def conversation(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
    ):
        conversation = self.conversation_manager.load_conversation(
            conversation_id=conversation_id, user_id=user_id
        )
        first_message = len(conversation) == 0

        if first_message:
            enriched_prompt = query
            if self.data_chain_driver.data_chains:  # Fix later.
                enriched_prompt = self.data_chain_driver.get_augmented_prompt(query)
            conversation = self.chat_prompt.format_messages(
                question=query, context=enriched_prompt
            )
            print(enriched_prompt, "*********" * 4)

        else:
            conversation.append(HumanMessage(content=query))
        try:
            ai_message = self.model.invoke(conversation)
            # save last conversations.
            self.conversation_manager.append_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                last_message=conversation[0],
            )
            self.conversation_manager.append_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                last_message=ai_message,
            )
            return {"role": "assistant", "content": ai_message.content}

        except Exception:
            logger.error(traceback.format_exc())
