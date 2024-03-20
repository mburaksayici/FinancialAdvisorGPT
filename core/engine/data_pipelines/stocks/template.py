import ast
import asyncio

from langchain import PromptTemplate

from base.base_datachain import AbstractDataChain
from core.engine.data_pipelines.stocks.stock_pipelines import StockPipeline
from core.engine.driver import ChainLLMModel

template = """I would like you to give me python list object by understanding the context. You are a financial analyst, so you can classify the financial data request.
I'll give a context (or question) that may include a stock data. Given a context, you'll give me a python list based on the rules. Don't answer anything except python list. Here are some rules.
1. If there's no request on a stock data, just return an empty list, [].
2. If there's a request on a stock data, you need to return a list of dictionaries, example: [{{"function":function_name, "symbol":company_symbol, "description":description}}].
3. function_name are OVERVIEW BALANCE_SHEET INCOME_STATEMENT EARNINGS IPO_CALENDAR WTI BRENT REAL_GDP REAL_GDP_PER_CAPITA. 
4. function_name is the classification I want from you. Here are the options ( word before the colon is the function name, word after the colon is the description of the function):

OVERVIEW : Company Overview, company information, financial ratios, and other key metrics for the equity specified. Data is generally refreshed on the same day a company reports its latest earnings and financials.
BALANCE_SHEET :   annual and quarterly balance sheets for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
INCOME_STATEMENT :  the annual and quarterly income statements for the company of interest, with normalized fields mapped to GAAP and IFRS taxonomies of the SEC. Data is generally refreshed on the same day a company reports its latest earnings and financials.
EARNINGS :  the annual and quarterly earnings (EPS) for the company of interest. Quarterly data also includes analyst estimates and surprise metrics.
IPO_CALENDAR :  a list of IPOs expected in the next 3 months.
WTI : Crude Oil Prices: West Texas Intermediate (WTI) :  the West Texas Intermediate (WTI) crude oil prices in daily, weekly, and monthly horizons. Source is  U.S. Energy Information Administration, Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma, retrieved from FRED, Federal Reserve Bank of St. Louis. This data feed uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis. 
BRENT :  the Brent (Europe) crude oil prices in daily, weekly, and monthly horizons. Source is U.S. Energy Information Administration, Crude Oil Prices: Brent - Europe, retrieved from FRED, Federal Reserve Bank of St. Louis.
REAL_GDP : the annual and quarterly Real GDP of the United States. Source: U.S. Bureau of Economic Analysis, Real Gross Domestic Product, retrieved from FRED, Federal Reserve Bank of St. Louis. 
REAL_GDP_PER_CAPITA : the quarterly Real GDP per Capita data of the United States.


5. If in the context, either there's a request to the data, or you think if the datas mentioned in 3 will be beneficial, you need to return a dictionary similar to [{{"function":function_name, "symbol":company_symbol}}]. If there's no request, you need to return an empty list, [].
One example, if the context is "I want to see the balance sheet of TSLA", you need to return [{{"function":"BALANCE_SHEET", "symbol":"TSLA", "description":"This data is ..., and it can be helpful on ..."}}]. However, if there are multiple requests, you need to return a list of dictionaries. For example, if the context is "I want to see the balance sheet of TSLA and the income statement of AAPL", you need to return [{{"function":"BALANCE_SHEET", "symbol":"TSLA", "description":"This data is ..., and it can be helpful on ..."}}, {{"function":"INCOME_STATEMENT", "symbol":"AAPL","description":"This data is ..., and it can be helpful on ..."}}].

Dont ever give any comment! Just give me the python list based on the rules! If you can't find anything, just return an empty list, [] !
You are only allowed to respond either [] or the Python list of dictionaries based on the rules! DONT ADD ANY COMMENT! JUST GIVE ME THE PYTHON LIST BASED ON THE RULES!
Direct answers only. No AI narrator or voice, AI is silent. No AI introduction. No AI summary. No disclaimers or warnings or usage advisories.

Here's the context : {context}
"""


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context"],
    template=template,
)


class StockDataChain(AbstractDataChain):
    """
    Class that allows RAG to retrieve online data.

    """

    def __init__(
        self, model: ChainLLMModel, prompt_template: PromptTemplate = PROMPT_TEMPLATE
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self.stock_pipeline = StockPipeline()

    def chat(self, context):
        return self.model.nonasync_chat(context, prompt_template=self.prompt_template)

    async def async_chat(self, context):
        return await self.model.async_chat(
            context, prompt_template=self.prompt_template
        )

    def get_data(self, context, return_augmented_prompt=True):
        response = self.chat(context)
        stock_pipeline_parameters = ast.literal_eval(response)
        print("searching this stock: ", stock_pipeline_parameters)
        data = list()

        for parameter in stock_pipeline_parameters[:1]:
            function = parameter["function"]
            symbol = parameter["symbol"]
            description = (
                parameter["description"] if "description" in parameter else None
            )

            response = self.stock_pipeline.query(function=function, symbol=symbol)
            data.append(
                {
                    "function": function,
                    "symbol": symbol,
                    "description": description,
                    "response": response,
                }
            )

        if return_augmented_prompt:

            augmented_prompt = """Here's the stock data I found for you. Please cite the resources with links and dates if it's given. Those data are provided from Alphavantage : """

            for stock in data:
                augmented_prompt += f"""I found the stock data below because {stock["description"]} . Those stock data are : \n"""
                augmented_prompt += f"""{str(stock["response"])}\n"""
            return augmented_prompt

        return data

    async def aget_data(self, context, return_augmented_prompt=True):
        response = await self.async_chat(context)
        stock_pipeline_parameters = ast.literal_eval(response)
        print("searching this stock: ", stock_pipeline_parameters)
        tasks = []

        for parameter in stock_pipeline_parameters:
            task = self.process_parameter(parameter)  # Keep the same
            tasks.append(task)  # Add await here

        results = await asyncio.gather(*tasks)

        if return_augmented_prompt:
            augmented_prompt = """Here's the stock data I found for you. Please cite the resources with links and dates if it's given. Those data are provided from Alphavantage : """

            for stock in results:
                augmented_prompt += f"""I found the stock data below because {stock["description"]} . Those stock data are : \n"""
                augmented_prompt += f"""{str(stock["response"])}\n"""
            return augmented_prompt

        return results

    async def process_parameter(self, parameter):
        function = parameter["function"]
        symbol = parameter["symbol"]
        description = parameter["description"]

        response = await self.stock_pipeline.aquery(function=function, symbol=symbol)

        return {
            "function": function,
            "symbol": symbol,
            "description": description,
            "response": response,
        }
