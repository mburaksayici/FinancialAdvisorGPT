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


dashboard_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
    You are a python tool that acts like growth analysts. You're serving to growth engineers.
    
    Given the data inside context, analyse the sales and give me four sentences in dict format.

    Assume this is the question: "How is the elidor shampoo sales doing?"

    Assume I'll provide the data that  you can analyse and answer the question. After analysis, your answer should be a python dictionary that:

    {{"headline": "Elidor sales are decreasing, beware!",
    "summary": "Elidor sales seems to be decreasing lately, due to bla bla bla.",
    "bullet_point_1": "In the last four months, elidor shampoo sales are decreasing.",
    "bullet_point_2": "Its share in hair care category is also decreasing although hair care category sales has increased %5.",
    "bullet_point_3": "Share in gold segments is increasing, but bronze segment sales has decreased significantly.",
    }} 

    Make the content of summary numerically rich. Support your claims with numerical data that you have.

    Here's the question : {question}

    Here's the context :  {context}



    Your Python dictionary answer  : 
    """,
        ),  # -->  SystemMessagePromptTemplate
        # MessagesPlaceholder(variable_name="history"),
    ]
)
