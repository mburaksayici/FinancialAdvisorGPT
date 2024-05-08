"""
News Pipelines that uses https://newsapi.org/docs/get-started

"""
import os
import requests
import pandas as pd

from logging_stack import logger
import asyncio

from core.analyse_engine.analyse import Analyse
from core.engine.text_utils.text_utils import text_similarity_finder


class GraphResultsRetrievalPipeline:
    def __init__(self):
        self.analyser = Analyse(pd.read_csv("data_category_added.csv"))

    def query(self, query, product_name=None):
        graph_outputs = dict()
        for query_graph in query:
            graph_name = query_graph["graph"]
            query_graph.pop("graph")
            graph_outputs[graph_name] = getattr(
                self.analyser,
                graph_name,
            )()
        return graph_outputs

    def aquery(self, query, product_name=None):
        graph_outputs = dict()
        for query_graph in query:
            graph_name = query_graph["graph"]
            query_graph.pop("graph")
            graph_outputs[graph_name] = getattr(
                self.analyser,
                graph_name,
            )(**query_graph)
        return graph_outputs


class ProductNameRetrievalPipeline:
    def __init__(self) -> None:
        self.product_names = (
            pd.read_csv("data_category_added.csv")["Description"].unique().tolist()
        )

    def initial_product_similarity_finder(self, word):
        similar_product_ranking = text_similarity_finder(word, self.product_names)
        return similar_product_ranking
