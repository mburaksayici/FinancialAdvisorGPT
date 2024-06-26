"""
Base class for datachain models
"""


from abc import ABC, abstractmethod


class AbstractDataChain(ABC):
    """
    Abstract base class for AbstractDataChain.
    """

    def __init__(self, model, prompt_template):
        self.model = model
        self.prompt_template = prompt_template

    @abstractmethod
    def chat(self, context):
        """
        Abstract method for generating a response from the language model given a context.

        Args:
        - context (str): The context for generating the response.

        Returns:
        - response (str): The response generated by the language model.
        """
        pass

    @abstractmethod
    def get_data(self, context):
        """
        Abstract method for retrieving data based on the response from the language model.

        Args:
        - context (str): The context for generating the response and querying data.

        Returns:
        - data (list): A list of dictionaries containing the retrieved data for each parameter.
        """
        pass
