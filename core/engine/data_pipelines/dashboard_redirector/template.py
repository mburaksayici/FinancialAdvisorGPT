import ast
import asyncio
from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.driver import ChainLLMModel

template = """You are a responsible for converting user comments to the json responses. I would like you to give me python list including dicts, by  understanding a context. 
Your answer must be python list, no other comments. Dont ever answer any comment. Just give me the python list of strings based on the rules! Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list that contains question strings.
Here's an  example : 
Context : "How are my elidor shampoo sales going?"
One possible answer that you can give is, choosing minimum 3 graphs.
Answer : [{{"graph": "top_sold_products", "since":"3m", }}, {{"graph":"category_share", "since":"3m"}}, {{"graph":"top_sold_products", "since":"3m"}} ]
Choose minimum 5 graphs. You need to return at least 5 elements in list. 

Here are the graph explanations, and parameters:

{{"graph":"category_share", "since":"3m"}} : Category share of the given product.   Since means monthly time.
{{"graph":"top_branches", "since":"3m"}} : Top branches, means a ranking of a franchise or shop of a company with respect to sales.  Since means monthly time.
{{"graph": "top_sold_products", "since":"3m"}} : Top sold products. Since means monthly time.
{{"graph": "monthly_sales", "since":"3m"}} : Monthly sales and revenue of a product. Since means monthly time.  "1m" means last one month. "1m-1m" means between 2 months ago and 1 month ago. "1m-1y" means last month of last year. 
{{"graph": "compare_monthly_sales", "since_list":["1m", "1m-1m"]}} : Monthly sales and revenue of a product including the option for analysing multiple dates. "1m" means last one month. "1m-1m" means between 2 months ago and 1 month ago. "1m-1y" means last month of last year. You can fill up since_list multiple ranges you would like to compare. 
{{"graph": "product_sales_chart", "frequency": "W"}} : Sales of a product and its category for 1 year, either for 52 weeks if frequency = "W", or for 12 months if frequency = "M"
{{"graph": "product_shares_comparison" }} : Share of a product among the category, for Last Month, Last Year Same Month and Year to Data.  There's no since options for that.
{{"graph": "product_shares" }} : Share of a product among the category, for last 3 months, each month separate. There's no since options for that.
{{"graph": "unique_customers" }} : Unique customers for given product, in "Year to Date", "Month to Date", and "Last Year Same Month".





Choose minimum 5 graphs. Parameters are upon your choice, you decide given the context, as a sales analyst or growth engineer.

Context : {context}
Answer: 

"""
# 5. If there's a date you want to search for, you can return the date, such as 2021-10-10. The date and time of the oldest article you want to get. If no date, don't place from to the dictionary.


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=template,
)


class GraphRetrievalChain(AbstractDataChain):
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

    def get_data(self, context, return_augmented_prompt=False):
        response = self.chat(context)
        dashboard_graph_kwargs = ast.literal_eval(response)
        return dashboard_graph_kwargs

    async def aget_data(self, context, return_augmented_prompt=True):
        response = self.chat(context)
        dashboard_graph_kwargs = ast.literal_eval(response)
        return dashboard_graph_kwargs


product_finder_template = """You are a responsible for finding product name given the conversation. I would like you to give me python list including dicts, by  understanding a context. 
Your answer must be a string, no other comments. Dont ever answer any comment. Just give me the python list of strings based on the rules! Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list that contains question strings.
Here's an  example : 
Start of Question : "How are my space owl sales going?" : End of Conversation
Start of Context : "SPACE OWL\nRETROSPOT LAMP\nTRAVEL SEWING KIT\nTOILET METAL SIGN\nFLOWERS TILE COASTER\nTEA TIME TEA TOWELS \nLED TEA LIGHTS\nRED RETROSPOT BOWL\nST TROPEZ NECKLACE\nDOILEY STORAGE TIN"
Then you'll look to the context I've sent you and you'll find out "Elidor Shining Shampoo" over there.
Your answer:  : ["SPACE OWL"]

Here are the product names:
Start of Conversation : {question} : End of Conversation
Start of Context : {context} : End of Context
Your answer: 
"""


PRODUCT_FINDER_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "context"],
    template=product_finder_template,
)


class ProductFinderChain(AbstractDataChain):
    """
    Class that allows finding product names.

    """

    def __init__(
        self,
        model: ChainLLMModel,
        prompt_template: PromptTemplate = PRODUCT_FINDER_PROMPT_TEMPLATE,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=False):
        return ast.literal_eval(self.chat(context))[0]

    async def aget_data(self, context, return_augmented_prompt=True):
        return ast.literal_eval(self.chat(context))[0]


product_finder_template = """You are a responsible for finding product name given the conversation. I would like you to give me python list including dicts, by  understanding a context. 
Your answer must be a string, no other comments. Dont ever answer any comment. Just give me the python list of strings based on the rules! Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list that contains question strings.
Here's an  example : 
Context : "How are my elidor shampoo sales going?"
Then you'll look to the context I've sent you and you'll find out "Elidor Shining Shampoo" over there.
Answer : ["Elidor Shampoo"]

Context : {context}
Your answer: 
"""


PRODUCT_GUESS_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=product_finder_template,
)


class ProductGuessChain(AbstractDataChain):
    """
    Class that allows finding product names.

    """

    def __init__(
        self,
        model: ChainLLMModel,
        prompt_template: PromptTemplate = PRODUCT_GUESS_PROMPT_TEMPLATE,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=False):

        product_name = ast.literal_eval(self.chat(context))
        if isinstance(product_name, list):
            return " ".join(product_name)
        if len(product_name) == 1:
            return product_name

    async def aget_data(self, context, return_augmented_prompt=True):
        return self.chat(context)
