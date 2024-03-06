"""
Stock pipelines from Alphavantage
"""


"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os

import requests
import aiohttp

# TO DO : Move to config.py


class StockPipeline:
    def __init__(self):
        self.api_key = os.getenv("STOCK_ALPHAVANTAGE_API_KEY")

    def query(self, function="OVERVIEW", symbol="TESL"):

        request_url = self.query_builder(function=function, symbol=symbol)

        response = requests.get(request_url)
        return response.json()

    async def aquery(self, function="OVERVIEW", symbol="TESL"):
        request_url = self.query_builder(function=function, symbol=symbol)

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(request_url) as response:
                return await response.json()

    def query_builder(self, function="OVERVIEW", symbol="TESL"):
        """
        Query builder for Alphavantage API

        Function : OVERVIEW, BALANCE_SHEET, INCOME_STATEMENT, EARNINGS, IPO_CALENDAR, WTI, REAL_GDP, TIME_SERIES_INTRADAY, TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, TIME_SERIES_MONTHLY,
        """
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={self.api_key}"
        return url
