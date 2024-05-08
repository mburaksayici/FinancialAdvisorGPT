import pandas as pd


class CSVIntegrator:

    EXPECTED_COLUMNS = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def read(
        self,
    ):
        self.df = pd.read_csv(
            self.file_path, parse_dates=["InvoiceDate"], infer_datetime_format=True
        )
        return self.df
