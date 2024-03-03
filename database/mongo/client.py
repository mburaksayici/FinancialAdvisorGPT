from mongoengine import connect
from config.config import MongoConfig
import models


# User DB
def connect_mongodb():
    connect(host=MongoConfig.DATABASE_URL)
