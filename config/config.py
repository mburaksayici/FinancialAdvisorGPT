from typing import Optional
import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import models as models

load_dotenv()


class Settings(BaseSettings):
    # database configurations
    DATABASE_URL: Optional[str] = None

    # JWT
    secret_key: str = "secret"
    algorithm: str = "HS256"

    class Config:
        env_file = ".env.dev"
        from_attributes = True


class MongoConfig:

    MONGODB_HOST = os.getenv(
        "MONGODB_HOST", "localhost"
    )  # ?authSource=admin&ssl=true&replicaSet=globaldb"
    MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")
    MONGO_URL = DATABASE_URL = f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}/finsean"


class RedisConfig:

    REDIS_HOST = os.getenv(
        "REDIS_HOST", "localhost"
    )  # ?authSource=admin&ssl=true&replicaSet=globaldb"
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")  # NOTE : Those configs can be merged.
