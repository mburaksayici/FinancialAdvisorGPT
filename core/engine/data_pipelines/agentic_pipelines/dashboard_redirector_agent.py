from typing import Optional, Annotated, Union
from core.engine.text_utils.text_utils import text_similarity_finder

from core.engine.data_pipelines.dashboard_redirector.dashboard_redicrector_pipelines import (
    GraphResultsRetrievalPipeline,
    ProductNameRetrievalPipeline,
)
from core.engine.data_pipelines.dashboard_redirector.template import (
    ProductGuessChain,
    ProductFinderChain,
)
from core.engine.data_pipelines.dashboard_redirector.template import GraphRetrievalChain
from core.engine.data_pipelines.internal_model_driver import InternalModelDriver
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.driver import ChainLLMModel


class DashboardRedirectorAgent:
    def __init__(self) -> None:
        self.internal_model_driver = InternalModelDriver()
        self.product_name_retrieval_pipeline = ProductNameRetrievalPipeline()
        self.model = ChainLLMModel(model="mistral-api")

        self.product_guess_chain = ProductGuessChain(self.model)
        self.product_finder_chain = ProductFinderChain(self.model)
        self.graph_retrieval_chain = GraphRetrievalChain(self.model)
        self.graph_result_retrieval_pipeline = GraphResultsRetrievalPipeline()

    def conversation(self, query) -> dict:
        # First guess the product name.
        product_guess_name = self.product_guess_chain.get_data(context=query)

        # Second, ai gets brand name
        similar_products = (
            self.product_name_retrieval_pipeline.initial_product_similarity_finder(
                product_guess_name
            )
        )

        # Third, brand name acquired by ai is searched through non-ai word similarity.
        shortlisted_similar_products = similar_products[
            : min(10, len(similar_products))
        ]

        # Fourth, shortlisted similar products
        shortlisted_similar_products = [i["word"] for i in shortlisted_similar_products]
        shortlisted_similar_products = "\n".join(shortlisted_similar_products)
        # Lastly, let AI decide what is the product name.
        product_name = self.product_finder_chain.get_data(
            {"question": query, "context": shortlisted_similar_products}
        )
        # Get the graph names
        graphs = self.graph_retrieval_chain.get_data(query)
        graphs_data = self.graph_result_retrieval_pipeline.query(
            graphs, product_name=product_name
        )
        return {"ai_data": graphs_data["ai_data"], "plot_data":graphs_data["plot_data"], "product_name": product_name}
