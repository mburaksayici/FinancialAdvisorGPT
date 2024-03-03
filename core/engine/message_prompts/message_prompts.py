from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
    I would like you to create a small report on a question. Here's the my  question for you : {question}
    Please use and cite the sources with the links/infos/urls/authors I've shared, Report should be based on the financial document I'm giving you between two *** . It is at the below.
***

 {context}

***

    There are other resources for you to use to answer the question : {question} .

    """,
        ),  # -->  SystemMessagePromptTemplate
        # MessagesPlaceholder(variable_name="history"),
    ]
)
