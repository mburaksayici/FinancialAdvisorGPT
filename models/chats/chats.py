from typing import Optional, Any, List

from mongoengine import ListField, StringField, Document


class Chats(Document):
    chat_id = StringField(max_length=200, required=True)
    user_id = StringField(max_length=200, required=True)
    message_content = ListField(required=True)

    meta = {"collection": "chats"}
