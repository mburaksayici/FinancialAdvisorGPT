import ast
import asyncio

from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.data_pipelines.news.news_pipelines import NewsPipeline
from core.engine.driver import ChainLLMModel
from core.engine.data_pipelines.summarizer.template import SummarizerChain

template = """I would like you to give me python list object, by  understanding a context. You are a financial analyst, so you can classify the financial data request.

You are only allowed to respond either [] or the Python list of dictionaries based on the rules!
Direct answers only. No AI narrator or voice, AI is silent. No AI introduction. No AI summary. No disclaimers or warnings or usage advisories.
 

News can have a significant impact on financial analysis and the financial markets in several ways:

Market Sentiment: News often influences investor sentiment, which can drive buying or selling activity in the markets. Positive news, such as strong earnings reports or favorable economic indicators, tends to boost investor confidence and lead to higher stock prices, while negative news, such as poor economic data or geopolitical tensions, can cause fear and uncertainty, resulting in market declines.
Price Movements: News events can directly affect the prices of financial assets, including stocks, bonds, currencies, and commodities. For example, a company announcing better-than-expected earnings may cause its stock price to rise, while news of a natural disaster affecting commodity production may lead to price spikes in the affected commodities.
Volatility: News can increase market volatility as investors react to new information. Sudden and unexpected news events, such as political developments or corporate scandals, can lead to sharp price fluctuations as investors reassess the impact of the news on asset valuations and market conditions.
Fundamental Analysis: News provides valuable information for fundamental analysis, which involves evaluating the financial health and performance of companies and economies. Analysts use news to update their earnings forecasts, assess industry trends, and make investment recommendations based on changes in business fundamentals.
Risk Management: News can impact risk management strategies by highlighting potential risks and opportunities in the markets. Traders and investors may adjust their portfolio allocations or implement hedging strategies in response to news events that could impact their investments.
Policy Decisions: News related to central bank decisions, government policies, and regulatory changes can have a significant impact on financial markets. For example, announcements of interest rate decisions or changes in monetary policy can influence borrowing costs, exchange rates, and asset prices.
Psychological Factors: News can also affect investor psychology and behavior, leading to herd mentality, irrational exuberance, or panic selling. Behavioral biases, such as anchoring or confirmation bias, may influence how investors interpret and react to news events, potentially amplifying market movements.

Here's one example of how news can impact financial analysis: Initial Vaccine Announcements (November 2020): When pharmaceutical companies like Pfizer and Moderna announced promising results from their COVID-19 vaccine trials, global financial markets responded positively. Stock markets rallied, with shares of companies in sectors most affected by the pandemic, such as travel, hospitality, and retail, experiencing significant gains. Investors became more optimistic about the prospect of a swift economic recovery as vaccination efforts ramped up.

I'll give a context (or question) that there may be news published related to the context, which can help in financial analysis. Given a context, you'll give me a python list based on the rules. Here are some rules.
1. If you think there's nothing to get from news, just return an empty list, []. For example, if the context is "I want to see the balance sheet of TSLA", you need to return []. I will search these kind of information by myself.
2. If there is, you need to return a list of dictionaries, example: [{{"query":query, "sortBy":sortBy, "country":country, "from":from_}}].
3. sortBy, The order to sort the articles in. Possible options: relevancy , popularity , publishedAt . relevancy = articles more closely related to q come first. popularity = articles from popular sources and publishers come first. publishedAt = newest articles come first.  For example, if you would like to get latest news, use publishedAt .
4. If there's a country that you want to search for, you can return the country, such as us. The 2-letter ISO 3166-1 code of the country you want to get headlines for. If no country, don't place country to the dictionary.
5. If there's a news search query that you would like to search, you need to set {{"query":query}}. It can be "query":"Apple" in python dict.
6. Please also add why you think the news is important for the context in one small sentence. For example, if the context is "I want to see if TESLA is able to produce batteries", you may to return [{{"query":"Lithium Stocks", "sortBy":"popularity", "country":"us", "description":"Tesla battery production is bounded to the lithium shortage."}}].
7. Response Example: [{{"query":"Apple", "sortBy":"publishedAt", "country":"us", "from":"2021-10-10"}}]
8. Response Example: [{{"query":"Tesla", "sortBy":"popularity", "country":"tr", "from":"2020-10-10"}}]

Dont ever answer any comment. Just give me the python list based on the rules! Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list of dictionaries based on the rules!
Direct answers only. No AI narrator or voice, AI is silent. No AI introduction. No AI summary. No disclaimers or warnings or usage advisories.


Here's the context : {context}
Python list response answer : 
"""
# 5. If there's a date you want to search for, you can return the date, such as 2021-10-10. The date and time of the oldest article you want to get. If no date, don't place from to the dictionary.


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=template,
)


class NewsDataChain(AbstractDataChain):
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(
        self, model: ChainLLMModel, prompt_template: PromptTemplate = PROMPT_TEMPLATE
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self.news_pipeline = NewsPipeline()
        self.summarizer_chain = SummarizerChain(model)

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=True):
        print("news obtaining")
        response = self.chat(context)
        news_pipeline_parameters = ast.literal_eval(response)
        print("searching this news: ", news_pipeline_parameters)

        data = list()
        for parameter in news_pipeline_parameters[0:1]:
            query = parameter["query"]
            sortBy = parameter["sortBy"]

            country = parameter["country"] if "country" in parameter else None
            description = (
                parameter["description"] if "description" in parameter else None
            )

            response = self.news_pipeline.query(
                query=query, sortBy=sortBy, country=country
            )
            data.append(
                {"query": query, "description": description, "response": response}
            )

        if return_augmented_prompt:

            if len(data) == 0:
                return "I couldn't find any news for you. Please try another context."
            augmented_prompt = """Here's the news I found for you. Please cite the resources with links and dates if it's given. : """

            for new in data[0:1]:
                if new is not None:
                    if (
                        new["response"]["totalResults"] != 0
                    ):  #  TO DO : 2 ifs  Will be fixed later
                        augmented_prompt += f"""I found the news below because {new["description"]} . Those news are : \n"""
                        for article in new["response"]["articles"][0:1]:
                            content = str(article["content"])
                            if len(content) > 750:  # Skip if content is too long
                                continue
                            if (
                                len(content) > 100
                            ):  # If exceeds 50 words (roughly 250 characters), summarize
                                content = self.summarizer_chain.get_data(
                                    context=content, word_count=20
                                )
                            augmented_prompt += f"""Date : {article["publishedAt"]} , url : {article["url"]} , content : {content}\n"""
            return augmented_prompt

        return data

    async def aget_data(self, context, return_augmented_prompt=True):
        print("news obtaining")
        response = await self.async_chat(context)
        news_pipeline_parameters = ast.literal_eval(response.replace("'", ""))
        print("searching this news: ", news_pipeline_parameters)

        data = []
        tasks = []

        for parameter in news_pipeline_parameters:
            query = parameter["query"]
            sortBy = parameter["sortBy"]
            country = parameter.get("country")
            description = parameter.get("description")

            task = await self.news_pipeline.aquery(
                query=query, sortBy=sortBy, country=country
            )
            tasks.append(task)

        # Use asyncio.gather to concurrently execute the tasks

        responses = await asyncio.gather(*tasks)
        print(responses)
        for parameter, response in zip(news_pipeline_parameters, responses):
            query = parameter["query"]
            description = parameter.get("description")
            data.append(
                {"query": query, "description": description, "response": response}
            )

        if return_augmented_prompt:

            augmented_prompts = []

            for new in data:
                if new is not None and new["response"]["totalResults"] != 0:
                    augmented_prompt = f"I found the news below because {new['description']} . Those news are : \n"
                    article_tasks = []
                    content_list = []
                    for article in new["response"]["articles"]:
                        content = str(article["content"])
                        if len(content) > 750:  # Skip if content is too long
                            content_list.append(content)
                        if len(content) > 100:  # If exceeds 100 characters, summarize
                            task = self.summarizer_chain.aget_data(
                                context=content, word_count=20
                            )
                            article_tasks.append(task)
                    print(article_tasks, "*********" * 10)
                    # Use asyncio.gather to concurrently execute the summarizer tasks for articles
                    content_summaries = await asyncio.gather(*article_tasks)
                    content_list.extend(content_summaries)
                    for article, summary in zip(
                        new["response"]["articles"], content_list
                    ):
                        augmented_prompt += f"Date : {article['publishedAt']} , url : {article['url']} , content : {summary}\n"

                    augmented_prompts.append(augmented_prompt)

            return "\n".join(augmented_prompts)

        return data
