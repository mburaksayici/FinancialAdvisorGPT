import datetime as dt
from dateutil.relativedelta import relativedelta
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
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])

        self.rfm_analysis = RFMAnalysis(self.df)
        self.cohort_analysis = CohortAnalysis(self.df)
        self.basket_analysis = BasketAprioriAnalysis(self.df)

    def filter_data(self, since=None, category=None, **kwargs):
        since_date = None

        filtered_df = self.df.copy()
        # Get the maximum date in the dataset
        max_date = pd.to_datetime(self.df["InvoiceDate"].max())
        if category:
            filtered_df = filtered_df[filtered_df["Category"] == category]
        if since is None:
            return filtered_df

        if "-" in since:
            # Parse the input string
            periods = since.split("-")
            main_period = periods[0]
            sub_period = periods[1]

            # Extract numeric values
            main_value = int(main_period[:-1])
            sub_value = int(sub_period[:-1])

            if main_period.endswith("m") and sub_period.endswith("y"):
                # Last X months of Y years ago
                reference_date = max_date - relativedelta(years=sub_value)
                since_date = reference_date - relativedelta(months=main_value)
            elif main_period.endswith("m") and sub_period.endswith("m"):
                # Last X months of Y months ago
                reference_date = max_date - relativedelta(months=sub_value)
                since_date = reference_date - relativedelta(months=main_value)
        else:
            # Single period case
            if since == "m":
                main_value = int(since[:-1])
                since_date = max_date - relativedelta(months=main_value)
            if since == "y":
                main_value = int(since[:-13])
                since_date = max_date - relativedelta(months=main_value)

        since = since_date if since_date else since

        # Filter data based on the calculated since_date
        filtered_df = filtered_df[filtered_df["InvoiceDate"] >= since]

        # Return the filtered dataframe
        return filtered_df

    def no_of_products(self, since="all", product_name="", **kwargs):
        if since == "all":
            return self.df["StockCode"].nunique()
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            data =  filtered_df["StockCode"].nunique()
            return {"ai_data": data, "plot_data" :  data }

    def no_of_customers(self, since="all", product_name="", **kwargs):
        if since == "all":
            data =  self.df["CustomerID"].nunique()
            return   {"ai_data": data, "plot_data" :  data }  
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            data = filtered_df["CustomerID"].nunique()
            return   {"ai_data": data, "plot_data" :  data }     

    def category_share(self, since="all", product_name="", **kwargs):
        if since == "all":
            total_unique_categories = self.df["Category"].nunique()
            total_categories = len(self.df["Category"])
            data =  total_unique_categories / total_categories

            return    {"ai_data": data, "plot_data" :  data }     
        else:
            filtered_df = self.filter_data(since)
            total_unique_categories = filtered_df["Category"].nunique()
            total_categories = len(filtered_df["Category"])
            data  =  total_unique_categories / total_categories
            return    {"ai_data": data, "plot_data" :  data }     


    def unique_customers(self, product_name, since="ytd", frequency="m"):
        since = "ytd"
        # Define the list to store results
        results = []
        plot_data = {
            "graph_type": "linear",
            "data": {"lines": [{"name": "Unique Customers", "x": [], "y": []}]},
            "headline": f"Unique Customers for {product_name}",
            "summary": f"Product: {product_name} - Unique Customers for selected period",
        }

        # Get the current max date in the data
        max_date = self.df["InvoiceDate"].max()

        # Define time periods based on `since` parameter
        if since == "ytd":
            start_date = pd.to_datetime(f"{max_date.year}-01-01")
        elif since == "mtd":
            start_date = pd.to_datetime(f"{max_date.year}-{max_date.month}-01")
        else:
            raise ValueError("Invalid 'since' parameter. Use 'ytd' or 'mtd'.")

        # Filter data for the specific period and product
        period_data = self.df[
            (self.df["InvoiceDate"] >= start_date)
            & (self.df["InvoiceDate"] <= max_date)
            & (self.df["Description"] == product_name)
        ]

        if frequency == "w":
            # Group by week
            period_data["Week"] = (
                period_data["InvoiceDate"]
                .dt.to_period("W")
                .apply(lambda r: r.start_time)
            )
            grouped_data = (
                period_data.groupby("Week")["CustomerID"].nunique().reset_index()
            )
            grouped_data.columns = ["Period", "UniqueCustomers"]
        elif frequency == "m":
            # Group by month
            period_data["Month"] = (
                period_data["InvoiceDate"]
                .dt.to_period("M")
                .apply(lambda r: r.start_time)
            )
            grouped_data = (
                period_data.groupby("Month")["CustomerID"].nunique().reset_index()
            )
            grouped_data.columns = ["Period", "UniqueCustomers"]
        else:
            raise ValueError(
                "Invalid 'frequency' parameter. Use 'w' for weekly or 'm' for monthly."
            )

        # Add data to results and plot_data
        for _, row in grouped_data.iterrows():
            result = {
                "period": row["Period"].strftime("%Y %B"),
                "product_name": product_name,
                "unique_customers": row["UniqueCustomers"],
            }
            results.append(result)
            plot_data["data"]["lines"][0]["x"].append(row["Period"].strftime("%Y %B"))
            plot_data["data"]["lines"][0]["y"].append(row["UniqueCustomers"])

        return {"ai_data": results, "plot_data": plot_data}
    
    


    def product_shares(self, product_name, **kwargs):
        # Define the list to store results
        results = []
        plot_data = {
            "graph_type": "pie",
            "data": {
                "pies": []
            },
            "headline": f"Share of Monthly Revenue for {product_name}",
            "summary": f"Product: {product_name} vs Others in Category"
        }

        # Obtain the category of the given product name
        product_category = self.df.loc[
            self.df["Description"] == product_name, "Category"
        ].iloc[0]

        # Get the current max date in the data
        max_date = self.df["InvoiceDate"].max()

        # Loop through the last three months
        for i in range(3):
            start_date = max_date - relativedelta(months=i + 1)
            end_date = max_date - relativedelta(months=i)

            # Filter data for the specific month and category
            monthly_data = self.df[
                (self.df["InvoiceDate"] >= start_date)
                & (self.df["InvoiceDate"] < end_date)
                & (self.df["Category"] == product_category)
            ]

            if monthly_data.empty:
                continue

            # Group by product and calculate the total sales
            product_sales = monthly_data.groupby("Description")["TotalPrice"].sum()

            # Calculate the total sales
            total_sales = product_sales.sum()

            # Get the share of the given product
            product_share = (
                (product_sales[product_name] / total_sales) * 100
                if product_name in product_sales
                else 0
            )
            others_share = 100 - product_share

            # Format the month
            month_formatted = start_date.strftime("%Y %B")

            # Store the result in a dictionary
            result = {
                "month": month_formatted,
                "product_name": product_name,
                "product_share": product_share,
                "others_share": others_share,
            }

            plot_data["data"]["pies"].append({
                "name": month_formatted,
                "x": ["Product", "Others"],
                "y": [product_share, others_share]
            })

            results.append(result)

        return {
            "ai_data": results,
            "plot_data": plot_data
        }

    def category_top_sold(self, since="all", category=None, **kwargs):
        return self.top_sold_products(since=since, category=category)

    """
    def top_branches(self, since="all", product_name="", n=5, **kwargs):
        if since == "all":
            return self.df["Country"].value_counts().head(n).to_dict()
        else:
            # Convert since to datetime format
            since_date = pd.to_datetime(since, format="%m/%d/%Y %H:%M")
            # Filter data based on the specified date
            filtered_df = self.df[self.df["InvoiceDate"] >= since_date]
            return filtered_df["Country"].value_counts().head(n).to_dict()
    """
    def top_sold_products(
        self, since="all", product_name="", n=5, category=None, **kwargs
    ):
        filtered_df = self.filter_data(since=since, category=category)

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

        return {"ai_data": sorted_products.to_dict(), "plot_data" : sorted_products.to_dict()  }

    def monthly_sales(self, since="1y", product_name="", **kwargs):
        if since == "all":
            filtered_df = self.df
        else:
            filtered_df = self.filter_data(since)
        if product_name != "":
            filtered_df = filtered_df[filtered_df["Description"] == product_name]

        # Convert 'InvoiceDate' column to datetime format
        # Extract the month from the 'InvoiceDate' column
        filtered_df['Month'] = filtered_df['InvoiceDate'].dt.strftime('%Y %B')

        # Group the data by month and count the number of items in each group
        items_per_month = filtered_df.groupby('Month').size()

        # Calculate the total amount spent for each item in each transaction
        filtered_df['TotalAmount'] = filtered_df['Quantity'] * filtered_df['UnitPrice']

        # Group the data by year and month, and sum the 'TotalAmount' column for each group
        monthly_revenue = filtered_df.groupby('Month')['TotalAmount'].sum().astype(int)
        monthly_quantity = filtered_df.groupby('Month')['Quantity'].sum().astype(int)

        # Ensure the data is ordered by month
        months = pd.date_range(start=filtered_df['InvoiceDate'].min(), end=filtered_df['InvoiceDate'].max(), freq='MS')
        months_str = months.strftime('%Y %B')

        # Fill in missing months with zero revenue and quantity
        monthly_revenue = monthly_revenue.reindex(months_str, fill_value=0)
        monthly_quantity = monthly_quantity.reindex(months_str, fill_value=0)

        # Prepare plot_data
        plot_data = {
            "graph_type": "linear",
            "data": {
                "lines": [
                    {"name": "Quantity", "x": months_str.tolist(), "y": monthly_quantity.values.tolist()},
                    {"name": "Revenue", "x": months_str.tolist(), "y": monthly_revenue.values.tolist()}
                ],
            },
            "headline": f"Monthly Revenue & Quantity of Product",
        }

        return {
            "ai_data": {
                "revenue": monthly_revenue.to_dict(),
                "sales_quantity": items_per_month.to_dict()
            },
            "plot_data": plot_data
        }    

    def compare_monthly_sales(
        self, since_list=["1m", "1m-1m"], product_name="", **kwargs
    ):
        agg_monthly_sales = list()
        for since in since_list:
            agg_monthly_sales.append(
                self.monthly_sales(since=since, product_name=product_name)
            )
        return agg_monthly_sales[0]

    def product_shares_comparison(self, product_name, since=None, **kwargs):
        # Define the list to store results
        results = []
        plot_data = {
                "graph_type": "pie",
                "data": {
                    "pies": [
                        
                    ]
                },
                "headline": f"Share of Monthly Revenue",
                "summary": f"Product: {product_name} vs Others in Category"
            }



        # Obtain the category of the given product name
        product_category = self.df.loc[
            self.df["Description"] == product_name, "Category"
        ].iloc[0]

        # Get the current max date in the data
        max_date = self.df["InvoiceDate"].max()

        # Define time periods
        periods = [
            ("Last Month", max_date - relativedelta(months=1), max_date),
            (
                "Last Year Same Month",
                max_date - relativedelta(years=1, months=1),
                max_date - relativedelta(years=1),
            ),
            ("Year to Date", pd.to_datetime(f"{max_date.year}-01-01"), max_date),
        ]

        # Loop through the defined periods
        for period_name, start_date, end_date in periods:
            # Filter data for the specific period and category
            period_data = self.df[
                (self.df["InvoiceDate"] >= start_date)
                & (self.df["InvoiceDate"] < end_date)
                & (self.df["Category"] == product_category)
            ]

            if period_data.empty:
                continue

            # Group by product and calculate the total sales
            product_sales = period_data.groupby("Description")["TotalPrice"].sum()

            # Calculate the total sales
            total_sales = product_sales.sum()

            # Get the share of the given product
            product_share = (
                (product_sales[product_name] / total_sales) * 100
                if product_name in product_sales
                else 0
            )
            others_share = 100 - product_share

            # Store the result in a dictionary
            result = {
                "period": period_name,
                "product_name": product_name,
                "product_share": product_share,
                "others_share": others_share,
            }

            plot_data["data"]["pies"].append({
                            "name": period_name,
                            "x": ["Product", "Others in Category"],
                            "y": [product_share, others_share]
                        })

            # Prepare plot_data for each period

            results.append(result)

        return {
                "ai_data": result,
                "plot_data": plot_data
            }

    def product_sales_chart(self, product_name="", frequency="W", since=None, **kwargs):
        # Define the period to look back (52 weeks or 12 months)
        if frequency == "W":
            num_periods = 52
            period_label = "Week"
        elif frequency == "M":
            num_periods = 12
            period_label = "Month"
        else:
            raise ValueError(
                "Invalid frequency. Use 'W' for weekly or 'M' for monthly."
            )

        # Get the current max date in the data
        max_date = self.df["InvoiceDate"].max()

        # Obtain the category of the given product name
        product_category = self.df.loc[
            self.df["Description"] == product_name, "Category"
        ].iloc[0]

        # Create a date range for the past periods (weeks or months)
        if frequency == "W":
            start_date = max_date - pd.DateOffset(weeks=num_periods)
        elif frequency == "M":
            start_date = max_date - pd.DateOffset(months=num_periods)

        # Filter the dataframe for the relevant time period and category
        filtered_df = self.df[
            (self.df["InvoiceDate"] >= start_date)
            & (self.df["Category"] == product_category)
        ]

        # Create a column for the period number (week or month)
        filtered_df[period_label] = (
            filtered_df["InvoiceDate"]
            .dt.to_period(frequency)
            .apply(lambda r: r.start_time)
        )

        # Group by period and description, then sum up the quantities
        period_sales = (
            filtered_df.groupby([period_label, "Description"])["Quantity"]
            .sum()
            .reset_index()
        )

        # Get period sales for the given product
        product_period_sales = period_sales[period_sales["Description"] == product_name]
        product_period_sales = product_period_sales[
            [period_label, "Quantity"]
        ].set_index(period_label)

        # Get period sales for the entire category
        category_period_sales = period_sales.groupby(period_label)["Quantity"].sum()

        # Ensure both series have the same index
        product_period_sales = product_period_sales.reindex(
            category_period_sales.index, fill_value=0
        )

        # Create a DataFrame with both series
        sales_df = pd.DataFrame(
            {
                "Product Sales": product_period_sales["Quantity"],
                "Category Sales": category_period_sales,
            }
        ).reset_index()

        # Format the index for display
        sales_df.index = sales_df[period_label].dt.strftime("%Y %B")

        # Prepare plot_data
        plot_data = {
            "graph_type": "linear",
            "data": {
                "lines": [
                    {
                        "name": "Product Sales",
                        "x": sales_df[period_label].dt.strftime("%Y %B").tolist(),
                        "y": sales_df["Product Sales"].tolist()
                    },
                    {
                        "name": "Category Sales",
                        "x": sales_df[period_label].dt.strftime("%Y %B").tolist(),
                        "y": sales_df["Category Sales"].tolist()
                    }
                ]
            },
            "headline": f"Sales of Product",
            "summary": f"{period_label}s Sales of Product: {product_name} vs. Category: {product_category}"
        }

        return {"ai_data": plot_data, "plot_data": plot_data}

    def average_basket_size(self, product_name, since="ytd"):
        since = "ytd"
        # Define the list to store results
        results = []
        plot_data = {
            "graph_type": "linear",
            "data": {"lines": [{"name": "Average Basket Size", "x": [], "y": []}]},
            "headline": f"Average Basket Size",
            "summary": f"Product: {product_name} - Average Basket Size for selected period",
        }

        # Get the current max date in the data
        max_date = self.df["InvoiceDate"].max()

        # Define time periods based on `since` parameter
        if since == "ytd":
            start_date = pd.to_datetime(f"{max_date.year}-01-01")
        elif since == "mtd":
            start_date = pd.to_datetime(f"{max_date.year}-{max_date.month}-01")
        else:
            raise ValueError("Invalid 'since' parameter. Use 'ytd' or 'mtd'.")

        # Filter data for the specific period and product
        period_data = self.df[
            (self.df["InvoiceDate"] >= start_date)
            & (self.df["InvoiceDate"] <= max_date)
            & (self.df["Description"] == product_name)
        ]

        # Group by month
        period_data["Month"] = (
            period_data["InvoiceDate"].dt.to_period("M").apply(lambda r: r.start_time)
        )
        grouped_data = (
            period_data.groupby("Month")
            .agg(Revenue=("UnitPrice", "sum"), Baskets=("InvoiceNo", "nunique"))
            .reset_index()
        )

        # Calculate average basket size
        grouped_data["AverageBasketSize"] = (
            grouped_data["Revenue"] / grouped_data["Baskets"]
        )

        # Add data to results and plot_data
        for _, row in grouped_data.iterrows():
            result = {
                "period": row["Month"].strftime("%Y-%m"),
                "product_name": product_name,
                "average_basket_size": row["AverageBasketSize"],
            }
            results.append(result)
            plot_data["data"]["lines"][0]["x"].append(row["Month"].strftime("%Y-%m"))
            plot_data["data"]["lines"][0]["y"].append(row["AverageBasketSize"])

        return {"ai_data": results, "plot_data": plot_data}

    def analyse(
        self,
    ):
        return {
            "top_sold_products": self.top_sold_products(),
            #  "top_branches": self.top_branches(),
            # "category_share":self.category_share(),
            "no_of_customers": self.no_of_customers(),
            "no_of_products": self.no_of_products(),
            "rfm_analysis": self.rfm_analysis.analyse(),
            "cohort_analysis": self.cohort_analysis.analyse(),
            "basket_analysis": self.basket_analysis.analyse(),
            "compare_monthly_sales": self.compare_monthly_sales(),
        }


df = CSVIntegrator(file_path="data_category_added.csv").read()

analyser = Analyse(df=df)
