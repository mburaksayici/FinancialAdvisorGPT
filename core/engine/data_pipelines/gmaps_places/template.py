import ast
import asyncio
from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.data_pipelines.gmaps_places.gmaps_places_pipelines import (
    GMapsPlacesRetrievalPipeline,
)

from core.engine.driver import ChainLLMModel
from core.engine.data_pipelines.summarizer.template import SummarizerChain

template = """You are a business analyst. I would like  to set up a new shop. Here's the competitors in the area I've chosen.: 

Context : {context}


Question: What do you think, should I set up a new shop there? 

"""


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=template,
)


class GMapsPlacesRetrievalChain(AbstractDataChain):
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(
        self, model: ChainLLMModel, prompt_template: PromptTemplate = PROMPT_TEMPLATE
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self.gmaps_places_retrieval_pipeline = GMapsPlacesRetrievalPipeline()
        self.summarizer_chain = SummarizerChain(model)

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    async def aget_data(self, context, return_augmented_prompt=True):

        nearby_places = self.gmaps_places_retrieval_pipeline.search_nearby_places(
            **context
        )
        if return_augmented_prompt:

            augmented_prompt = f"""Here's the nearby locations in {context["sector"]} domain I found for you to help you to answer the question: """
            for i, place in enumerate(nearby_places):  # TO DO : Uncover or fix limit
                augmented_prompt += f""" {i}. Place.  Name : {place["name"]} . Stars : {place["stars"]} . comments : {str(place["comments"])} .   \n"""
            return augmented_prompt

    async def aget_raw_data(self, context):

        return self.gmaps_places_retrieval_pipeline.search_nearby_places(**context)

    def get_data(self, context, return_augmented_prompt=True):

        nearby_places = self.gmaps_places_retrieval_pipeline.search_nearby_places(
            **context
        )
        if return_augmented_prompt:

            augmented_prompt = f"""Here's the nearby locations in {context["sector"]} domain I found for you to help you to answer the question: """
            for i, place in enumerate(nearby_places):  # TO DO : Uncover or fix limit
                augmented_prompt += f""" {i}. Place.  Name : {place["name"]} . Stars : {place["stars"]} . comments : {str(place["comments"])} .   \n"""
            return augmented_prompt

    def get_raw_data(self, context, **kwargs):

        return self.gmaps_places_retrieval_pipeline.search_nearby_places(**context)
