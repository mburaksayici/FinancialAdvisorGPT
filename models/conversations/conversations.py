from typing import Optional, Any, List

from mongoengine import ListField, StringField, Document


class Conversations(Document):
    conversation_id = StringField(max_length=200, required=True)
    user_id = StringField(max_length=200, required=True)
    message_content = ListField(required=False)

    meta = {"collection": "conversations"}
