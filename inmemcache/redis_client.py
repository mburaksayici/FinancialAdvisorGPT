import redis
import json

from core.config import RedisConfig


class RedisClient:
    def __init__(self, host="localhost", port=6379, db=0, max_connections=10):
        self.redis_pool = redis.ConnectionPool(
            host=RedisConfig.REDIS_HOST,
            port=RedisConfig.REDIS_PORT,
            db=0,
            max_connections=20,
        )

    def get_connection(self):
        return redis.Redis(connection_pool=self.redis_pool)


class RedisManager:
    def __init__(self, redis_client):
        self.redis_client = RedisClient()

    def save_data(self, key, data):
        try:
            conn = self.redis_client.get_connection()
            conn.set(key, json.dumps(data))
        except redis.RedisError as e:
            # Handle Redis errors gracefully
            print(f"Error saving data to Redis: {e}")

    def load_data(self, key):
        try:
            conn = self.redis_client.get_connection()
            data = conn.get(key)
            if data:
                return json.loads(data)
            else:
                return None
        except redis.RedisError as e:
            # Handle Redis errors gracefully
            print(f"Error loading data from Redis: {e}")
