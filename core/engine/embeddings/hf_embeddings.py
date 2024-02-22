# TO DO : EMBEDDING BASECLASS OR TEMPLATE OR INTERFACE TO BE ADDED.

from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings()  # GPT4AllEmbeddings()


class HFEmbeddings:  # TO DO : BASECLASS OR TEMPLATE OR INTERFACE TO BE ADDED.
    def __init__(self, **kwargs) -> None:
        pass

    def embed(self, text, **kwargs):
        return embedding.embed(text, **kwargs)
