"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os

import requests

news_api_key = os.getenv(
    "NEWS_API_KEY"
)  ## NOTE : This is for persistend db storage. Clients with serving to be added.


class NewsPipeline:
    def __init__(self):
        self.api_key = news_api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def query(self, query, sortBy="popularity", country="us", from_=None):
        params = {
            "q": query,
            "country": country,
            "sortBy": sortBy,
            "pageSize": 5,
            "apiKey": self.api_key,
        }
        # if from_:
        #    params['from'] = from_

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for any HTTP error
            return response.json()
        except requests.RequestException as e:
            print(f"Error occurred: {e}")
            return None
