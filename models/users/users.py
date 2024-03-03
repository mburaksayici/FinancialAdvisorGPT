from typing import Optional, Any

from mongoengine import ListField, StringField, Document, ObjectIdField
from bson.objectid import ObjectId


class Users(Document):
    # user_id =  ObjectIdField(required=True, default=ObjectId,
    #                unique=True, primary_key=True)

    user_name = StringField(max_length=200, required=True)
    user_password = StringField(
        max_length=200, required=False
    )  # those are all prototype right now.

    meta = {"collection": "users"}
