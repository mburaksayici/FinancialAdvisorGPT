"""
Conversation manager 
"""
from typing import List
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from models import Conversations


class ConversationManager:
    def __init__(
        self,
    ) -> None:
        pass  # to be filled later

    def initialise_conversation(self, user_id: str):
        conversation_id = str(uuid.uuid4())
        Conversations(
            conversation_id=conversation_id, user_id=user_id, message_content=[]
        ).save()
        return conversation_id

    def save_conversation(self, conversation_id: str, user_id: str, conversation: List):
        message_content = list()
        for message in conversation:
            message_content.append({"type": message.type, "content": message.content})
        Conversations(
            conversation_id=conversation_id,
            user_id=user_id,
            message_content=message_content,
        ).save()
        return conversation_id

    def append_to_conversation(self, conversation_id: str, user_id: str, last_message):
        conversation_doc = Conversations.objects.get(
            conversation_id=conversation_id, user_id=user_id
        )
        last_message_dict = {"type": last_message.type, "content": last_message.content}
        conversation_doc.update(push__message_content=last_message_dict)

    def load_conversation(self, conversation_id, user_id):
        conversation_doc = Conversations.objects.get(
            conversation_id=conversation_id, user_id=user_id
        )
        conversation = list()
        for message in conversation_doc.message_content:
            # TO DO : Reduce ifs by dynamicstring and getattr
            if message["type"] == "human":
                conversation.append(HumanMessage(message["content"]))
            if message["type"] == "ai":
                conversation.append(AIMessage(message["content"]))
            if message["type"] == "system":
                conversation.append(SystemMessage(message["content"]))
        return conversation
