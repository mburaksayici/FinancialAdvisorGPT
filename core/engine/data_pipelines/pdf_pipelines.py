# TO DO : This is very ugly pipeline. It'll be fixed.

from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader


def pdf_loader_factory(loader="PyPDFLoader"):
    if loader == "PyPDFLoader":
        return PyPDFLoader
    elif loader == "OnlinePDFLoader":
        return OnlinePDFLoader
