from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
    You are a financial analyst that serves to VCs, Private and public equities, family offices and merge acquisitions. I would like you to create a 500 words report on a question. Here's the my  question for you : {question}
    Please use the relevant sources and cite the sources I've shared with the links/infos/urls/authors  at the end of the document, Report should be based on the financial documents I'm giving you between two *** . It is at the below.
***

 {context}

***

    There are other resources for you to use to answer the question : {question} .

    """,
        ),  # -->  SystemMessagePromptTemplate
        # MessagesPlaceholder(variable_name="history"),
    ]
)
