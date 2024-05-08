"""
Used to run agents that run internally.
"""

from logging_stack import logger
import traceback

from langchain.embeddings import HuggingFaceEmbeddings

from core.engine.message_prompts.message_prompts import chat_prompt
from core.engine.data_chain_driver import DataChainDriver
from models import *

# from core.engine.llm_models.mistral_7b_engine import Mistral7BInstructModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class InternalModelDriver:
    def __init__(self) -> None:
        self.model = None
        self.data_chains = None
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later
        self.chat_prompt = chat_prompt
        self.data_chain_driver = DataChainDriver()
        self.text_splitter = text_splitter_factory(
            text_splitter="RecursiveCharacterTextSplitter"
        )
        self.embedding = HuggingFaceEmbeddings()

    def set_data_chains(self, data_chains: list):
        self.data_chain_driver.set_data_chains(data_chains)

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def load_model(self, model_name):  # TO DO : Move model DB to mongo.
        # if model_name == "mistral":
        #    self.model = Mistral7BInstructModel(
        #        ""
        #    ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.
        if model_name == "mistral-api":
            self.model = MistralAPIModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.

    def conversation(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
    ):
        enriched_prompt = self.data_chain_driver.get_augmented_prompt(query)
        conversation = self.chat_prompt.format_messages(
            question=query, context=enriched_prompt
        )
        try:
            ai_message = self.model.invoke(conversation)
            # save last conversations.
            return {"role": "assistant", "content": ai_message.content}

        except Exception:
            logger.error(traceback.format_exc())
