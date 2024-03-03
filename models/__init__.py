from models.chats.chats import Chats
from models.users.users import Users

# word after models, defines db name. collection names are defined in inner class of document models, see the class inside User class

__all__ = [Users, Chats]
