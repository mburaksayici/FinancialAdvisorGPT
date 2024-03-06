import ast

from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.driver import ChainLLMModel

template = """Please summarize the given context to roughly {word_count} words. Please only give me the summary, no other comments. Dont ever answer any comment.
Context : {context}

"""
# 5. If there's a date you want to search for, you can return the date, such as 2021-10-10. The date and time of the oldest article you want to get. If no date, don't place from to the dictionary.


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["word_count", "context"],
    template=template,
)


class SummarizerChain(AbstractDataChain):
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(
        self, model: ChainLLMModel, prompt_template: PromptTemplate = PROMPT_TEMPLATE
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=True, word_count: int = 50):
        print("summarizing")
        return self.chat({"word_count": word_count, "context": context})

    async def aget_data(
        self, context, return_augmented_prompt=True, word_count: int = 50
    ):
        print("summarizing")
        return await self.async_chat({"word_count": word_count, "context": context})
