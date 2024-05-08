import datetime as dt
from operator import attrgetter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
from networkx.readwrite import json_graph

from core.analyse_engine.csv_integrator import CSVIntegrator


class BasketAprioriAnalysis:
    """
    Reference : https://www.kaggle.com/code/yugagrawal95/market-basket-analysis-apriori-in-python/notebook
    https://deepnote.com/app/code-along-tutorials/Market-Basket-Analysis-in-Python-An-Implementation-with-Online-Retail-Data-6231620b-cba3-4935-bde8-8ce1490868bf
    """

    @staticmethod
    def convert_into_binary(x):
        if x > 0:
            return 1
        else:
            return 0

    def __init__(self, df) -> None:
        self.df = df.copy()

    def _preprocess_data(
        self,
    ):
        self.df["Description"] = self.df["Description"].str.strip()
        self.df = self.df[self.df.Quantity > 0]

        self.basket = pd.pivot_table(
            data=self.df,
            index="InvoiceNo",
            columns="Description",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
        )
        # basket_sets = self.basket.applymap(BasketAprioriAnalysis.convert_into_binary)

        basket_sets = self.basket.map(lambda x: 1 if x > 0 else 0)
        basket_sets.drop(columns=["POSTAGE"], inplace=True)
        # call apriori function and pass minimum support here we are passing 7%. means 7 times in total number of transaction that item was present.
        frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
        return frequent_itemsets

    def _draw_graph(self, rules, no_of_rules_to_show):
        G1 = nx.DiGraph()
        color_map = []
        N = 50
        colors = np.random.rand(N)
        strs = [
            "R0",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R6",
            "R7",
            "R8",
            "R9",
            "R10",
            "R11",
        ]

        for i in range(no_of_rules_to_show):
            G1.add_nodes_from(["R" + str(i)])

            for a in rules.iloc[i]["antecedents"]:
                G1.add_nodes_from([a])
                G1.add_edge(a, "R" + str(i), color=colors[i], weight=2)
            for c in rules.iloc[i]["consequents"]:
                G1.add_nodes_from([c])
                G1.add_edge("R" + str(i), c, color=colors[i], weight=2)

        for node in G1:
            found_a_string = False
            for item in strs:
                if node == item:
                    found_a_string = True
            if found_a_string:
                color_map.append("yellow")
            else:
                color_map.append("green")

        edges = G1.edges()
        colors = [G1[u][v]["color"] for u, v in edges]
        weights = [G1[u][v]["weight"] for u, v in edges]

        pos = nx.spring_layout(G1, k=16, scale=1)
        nx.draw(
            G1,
            pos,
        )  #   edgeList=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16,    with_labels=False

        for p in pos:  # raise text positions
            pos[p][1] += 0.07
            nx.draw_networkx_labels(G1, pos)
            plt.show()

    def analyse(self):
        frequent_itemsets = self._preprocess_data()

        rules_mlxtend = association_rules(
            frequent_itemsets, metric="lift", min_threshold=0
        )

        rules_mlxtend = rules_mlxtend[
            (rules_mlxtend["lift"] >= 4) & (rules_mlxtend["confidence"] >= 0.8)
        ]  # rules_mlxtend['lift'] >= 4) & (rules_mlxtend['confidence'] >= 0.8)
        # self._draw_graph(rules_mlxtend, 10 )
        return rules_mlxtend


class CohortAnalysis:
    def __init__(self, df):
        self.df = df.copy()

    def analyse(self):

        data = self.df[["CustomerID", "InvoiceNo", "InvoiceDate"]].drop_duplicates()
        data["order_month"] = data["InvoiceDate"].dt.to_period("M")
        data["cohort"] = (
            data.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
        )
        cohort_data = (
            data.groupby(["cohort", "order_month"])
            .agg(n_customers=("CustomerID", "nunique"))
            .reset_index(drop=False)
        )
        cohort_data["period_number"] = (
            cohort_data.order_month - cohort_data.cohort
        ).apply(attrgetter("n"))
        cohort_pivot = cohort_data.pivot_table(
            index="cohort", columns="period_number", values="n_customers"
        )
        cohort_size = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
        return retention_matrix.to_json()


class RFMAnalysis:
    def __init__(self, df) -> None:
        self.df = df

        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

    def _get_rfm_scores(self, df) -> pd.core.frame.DataFrame:

        df_ = df.copy()
        df_["recency_score"] = pd.qcut(df_["recency"], 5, labels=[5, 4, 3, 2, 1])
        df_["frequency_score"] = pd.qcut(
            df_["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )
        df_["monetary_score"] = pd.qcut(df_["monetary"], 5, labels=[1, 2, 3, 4, 5])
        df_["RFM_SCORE"] = df_["recency_score"].astype(str) + df_[
            "frequency_score"
        ].astype(str)

        return df_

    def _segment(self, rfm):

        seg_map = {
            r"[1-2][1-2]": "hibernating",
            r"[1-2][3-4]": "at_Risk",
            r"[1-2]5": "cant_loose",
            r"3[1-2]": "about_to_sleep",
            r"33": "need_attention",
            r"[3-4][4-5]": "loyal_customers",
            r"41": "promising",
            r"51": "new_customers",
            r"[4-5][2-3]": "potential_loyalists",
            r"5[4-5]": "champions",
        }

        rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
        return rfm["segment"].value_counts().sort_values(ascending=False)

    def analyse(
        self,
    ):

        today_date = dt.datetime(2011, 12, 11)

        rfm = self.df.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (today_date - x.max()).days,
                "InvoiceNo": lambda x: x.nunique(),
                "TotalPrice": lambda x: x.sum(),
            }
        )

        rfm.columns = ["recency", "frequency", "monetary"]
        # rfm['monetary'] = rfm[rfm['monetary'] > 0]
        rfm = rfm.reset_index()

        rfm = self._get_rfm_scores(rfm)

        return self._segment(rfm=rfm).to_dict()


class Analyse:
    def __init__(self, df) -> None:
        self.df = df
        self.rfm_analysis = RFMAnalysis(self.df)
        self.cohort_analysis = CohortAnalysis(self.df)
        self.basket_analysis = BasketAprioriAnalysis(self.df)

    def filter_data(self, since):
        # Get the maximum date in the dataset
        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"], format="ISO8601"
        )

        max_date = pd.to_datetime(self.df["InvoiceDate"].max())

        # Define the offset based on the input string
        offset = pd.DateOffset(months=int(since[:-1]))

        # Calculate the since date by subtracting the offset from the maximum date
        since_date = max_date - offset

        # Filter data based on the specified date
        filtered_df = self.df[self.df["InvoiceDate"] >= since_date]

        # Return the number of unique StockCodes in the filtered dataframe
        return filtered_df["StockCode"].nunique()

    def no_of_products(self, since="all", product_name=""):
        if since == "all":
            return self.df["StockCode"].nunique()
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            return filtered_df["StockCode"].nunique()

    def no_of_customers(self, since="all", product_name=""):
        if since == "all":
            return self.df["CustomerID"].nunique()
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            return filtered_df["CustomerID"].nunique()

    def category_share(self, since="all", product_name=""):
        if since == "all":
            total_unique_categories = self.df["Category"].nunique()
            total_categories = len(self.df["Category"])
            return total_unique_categories / total_categories
        else:
            filtered_df = self.filter_data(since)
            total_unique_categories = filtered_df["Category"].nunique()
            total_categories = len(filtered_df["Category"])
            return total_unique_categories / total_categories

    def top_branches(self, since="all", product_name="", n=5):
        if since == "all":
            return self.df["Country"].value_counts().head(n).to_dict()
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            return filtered_df["Country"].value_counts().head(n).to_dict()

    def top_sold_products(self, since="all", product_name="", n=5):
        if since == "all":
            filtered_df = self.df
        else:
            filtered_df = self.filter_data(since)

        # Count occurrences of each product
        product_counts = filtered_df["Description"].value_counts()

        # Sum of UnitPrice for each product
        unit_price_sum = filtered_df.groupby("Description")["UnitPrice"].sum()

        # Combine counts and sums
        product_data = pd.DataFrame(
            {"Counts": product_counts, "UnitPriceSum": unit_price_sum}
        )

        # Sort by counts
        sorted_products = product_data.sort_values(by="Counts", ascending=False).head(n)

        return sorted_products.to_dict()

    def analyse(
        self,
    ):
        return {
            "top_sold_products": self.top_sold_products(),
            "top_branches": self.top_branches(),
            # "category_share":self.category_share(),
            "no_of_customers": self.no_of_customers(),
            "no_of_products": self.no_of_products(),
            "rfm_analysis": self.rfm_analysis.analyse(),
            "cohort_analysis": self.cohort_analysis.analyse(),
            "basket_analysis": self.basket_analysis.analyse(),
        }


df = CSVIntegrator(file_path="datacleaned.csv").read()

analyser = Analyse(df=df)
