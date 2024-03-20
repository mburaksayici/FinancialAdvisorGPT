import os

# from mistralai.async_client import MistralAsyncClient
# from mistralai.models.chat_completion import ChatMessage
from langchain_mistralai.chat_models import ChatMistralAI

mistral_api_key = os.getenv("MISTRAL_API_KEY")


class MistralAPIModel:
    def __init__(
        self, model_path
    ) -> None:  # TO DO : Model configs can be jsonable later on for distribution.
        self.model_path = model_path
        print("using api model")

    def load_model(self):
        print("using api model")
        return ChatMistralAI(
            mistral_api_key=mistral_api_key, model="mistral-large-latest"
        )
