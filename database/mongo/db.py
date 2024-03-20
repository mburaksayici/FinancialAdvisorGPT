"""Mongodb functions to crud data from"""
from models import Users, Conversations


class DBUtils:
    def write_chat(chat_history, conversation_id, user_id):
        Conversations(
            chat_history=chat_history, conversation_id=conversation_id, user_id=user_id
        ).save()

    def read_check(conversation_id):
        return Conversations.objects(conversation_id=conversation_id)

    def write_user(user_name, user_password):
        Users(
            user_name=user_name,
            user_password=user_password,
        ).save()

    def read_user(user_name):
        return Users.objects(user_name=user_name)
