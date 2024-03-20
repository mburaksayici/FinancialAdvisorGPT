import ast
import asyncio
from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.data_pipelines.db_retrieval.db_retrieval_pipelines import (
    DBRetrievalPipeline,
)
from core.engine.driver import ChainLLMModel
from core.engine.data_pipelines.summarizer.template import SummarizerChain

template = """You are a financial analyst. I would like you to give me python list object, by  understanding a context. Python list object will include 20 questions where the context can be more understood if 5 questions are answered. 
Your answer must be python list, no other comments. Dont ever answer any comment. Just give me the python list of strings based on the rules! Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list that contains question strings.
Here's an  example : 
Context : "What is the financial status of the company?"
Answer : ["What is the net income of the company?", "What is the revenue of the company?", "What is the profit of the company?", "What is the loss of the company?", "What is the cash flow of the company?", "What is the debt of the company?", "What is the equity of the company?", "What is the assets of the company?", "What is the liabilities of the company?", "What is the operating income of the company?", "What is the gross profit of the company?", "What is the EBITDA of the company?", "What is the EBIT of the company?", "What is the EPS of the company?", "What is the PE ratio of the company?", "What is the PB ratio of the company?", "What is the PS ratio of the company?", "What is the ROE of the company?", "What is the ROA of the company?", "What is the ROIC of the company?"]

Return me a 20 questions in python list of strings given below.

Context : {context}

"""
# 5. If there's a date you want to search for, you can return the date, such as 2021-10-10. The date and time of the oldest article you want to get. If no date, don't place from to the dictionary.


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=template,
)


class DBRetrievalChain(AbstractDataChain):
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(
        self, model: ChainLLMModel, prompt_template: PromptTemplate = PROMPT_TEMPLATE
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self.db_retrieval_pipeline = DBRetrievalPipeline()
        self.summarizer_chain = SummarizerChain(model)

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=True):
        response = self.chat(context)
        print("retrieving documents")
        db_retrieval_queries = ast.literal_eval(response)

        data = relevant_docs_list = list()

        for query in db_retrieval_queries:

            response = self.db_retrieval_pipeline.query(query=query, k=5)

            for resp in response:
                resp["query"] = query

            relevant_docs_list.extend(response)

        if return_augmented_prompt:

            augmented_prompt = """Here's the data sources I found for you to help you to answer the question. You may or may not prefer to consider all questions and answers. If you think questions are relevant, and context is relevant to the question, you can create an answer by referencing the facts from the documents.   Please cite the resources with links and dates if it's given. : """
            for i, db_source in enumerate(
                relevant_docs_list[0:4]
            ):  # TO DO : Uncover or fix limit
                content = db_source["page_content"]
                if len(content) > 100:
                    content = self.summarizer_chain.get_data(content, word_count=20)
                augmented_prompt += f"""{i}. Question : {db_source["query"]} . filename : {db_source["source"]} . Context : {content} \n"""

            return augmented_prompt

        return data

    async def aget_data(self, context, return_augmented_prompt=True):
        response = await self.async_chat(context)
        print("retrieving documents")
        db_retrieval_queries = ast.literal_eval(response)

        data = relevant_docs_list = list()

        tasks = []
        for query in db_retrieval_queries:
            task = self.db_retrieval_pipeline.aquery(query=query, k=2)
            tasks.append(task)

        relevant_docs_lists = await asyncio.gather(*tasks)
        relevant_docs_list = [doc for sublist in relevant_docs_lists for doc in sublist]

        if return_augmented_prompt:
            augmented_prompt = """Here's the data sources I found for you to help you to answer the question. You may or may not prefer to consider all questions and answers. If you think questions are relevant, and context is relevant to the question, you can create an answer by referencing the facts from the documents. Please cite the resources with links and dates if it's given. : """

            # Concurrently gather content for each relevant document
            acontent_tasks = []
            content_results = []

            for db_source in relevant_docs_list[0:1]:  # TO DO : Uncover the limit
                content = db_source["page_content"]
                if len(content) > 100:
                    task = self.summarizer_chain.aget_data(content, word_count=20)
                    acontent_tasks.append(task)
                else:
                    # If content is less than 100 characters, append it directly
                    content_results.append(content)

            acontent_results = await asyncio.gather(
                *acontent_tasks, return_exceptions=True
            )
            content_results.extend(acontent_results)

            # Combine the content with other information and construct the augmented prompt
            for i, (db_source, content) in enumerate(
                zip(relevant_docs_list, content_results)
            ):
                if isinstance(content, Exception):
                    # Handle exceptions if any occurred during gathering
                    augmented_prompt += f"Error fetching content for document {i}: {type(content).__name__}\n"
                else:
                    augmented_prompt += f"{i}. Question : {db_source['query']} . filename : {db_source['source']} . Context : {content} \n"

            return augmented_prompt

        return data
