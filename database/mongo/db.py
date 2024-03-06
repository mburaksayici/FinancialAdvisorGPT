"""Mongodb functions to crud data from"""
from models import Users, Chats


class DBUtils:
    def write_chat(chat_history, chat_id, user_id):
        Chats(chat_history=chat_history, chat_id=chat_id, user_id=user_id).save()

    def read_check(chat_id):
        return Chats.objects(chat_id=chat_id)

    def write_user(user_name, user_password):
        Users(
            user_name=user_name,
            user_password=user_password,
        ).save()

    def read_user(user_name):
        return Users.objects(user_name=user_name)
