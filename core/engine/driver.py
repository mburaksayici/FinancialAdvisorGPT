import logging
from queue import Empty
from threading import Thread

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

from core.engine.data_pipelines.pdf_pipelines import pdf_loader_factory
from core.engine.llm_models.mistral_7b_engine import Mistral7BInstructModel
from core.engine.llm_models.mistral_api_engine import MistralAPIModel
from core.engine.text_utils.text_utils import text_splitter_factory


class ModelDriver:
    # Prompt
    TEMPLATE = """
    I would like you to create a detailed report. Report should be based on the financial document I'm giving you between two *** . It is at the below.


***

 {context}

***


I would like the report to answer the questions below, between two ---.

---

Financial Performance and Metrics:

What are Company's revenue trends over the past few years?
How has Company's gross margin changed over time?
What are the major components of Company's cost of goods sold?
What is Company's operating margin, and how does it compare to industry peers?
How has Company's net income evolved quarter-over-quarter and year-over-year?
What is Company's cash flow from operations, and is it growing or declining?
What are Company's key profitability ratios, such as return on assets and return on equity?
How does Company's debt level compare to its equity, and what is its debt-to-equity ratio?
What are Company's liquidity ratios, such as the current ratio and quick ratio?
How has Company's stock price performed relative to its financial performance?
Business Operations:
11. What are Company's primary business segments, and what percentage of revenue does each contribute?

How many vehicles has Company produced and delivered in recent quarters?
What are Company's plans for expanding production capacity?
How does Company distribute its products, and what are its sales channels?
What are Company's main markets geographically, and how does it plan to expand internationally?
What are Company's main sources of revenue, and are there any significant shifts in revenue streams?
What is Company's strategy for research and development, and how much does it invest in R&D?
What are Company's competitive advantages and key differentiators?
What are the main risks to Company's business, including regulatory, competitive, and technological risks?
How does Company manage supply chain risks, including sourcing of raw materials and components?
Financial Position and Balance Sheet:
21. What are Company's total assets, liabilities, and shareholder equity?

How much cash and cash equivalents does Company have on hand?
What are Company's long-term investments and their fair value?
What is the composition of Company's inventory, and how does it compare to previous periods?
What are Company's accounts receivable and accounts payable balances?
What are Company's long-term debt obligations, including principal amounts and interest rates?
How much stock-based compensation does Company expense each year?
What are Company's accrued liabilities and other non-current liabilities?
How does Company account for warranty reserves and other provisions?
What is Company's effective tax rate, and how does it impact its financial statements?
Financial Risk Management:
31. What are Company's hedging strategies for managing currency and commodity price risks?

How does Company manage interest rate risks associated with its debt?
What are Company's policies for managing credit risks related to customers and suppliers?
How does Company assess and mitigate risks associated with changes in regulatory environments?
What insurance coverage does Company maintain to protect against various risks?
How does Company evaluate and manage risks related to cybersecurity and data privacy?
What are Company's disaster recovery and business continuity plans?
How does Company monitor and manage risks related to environmental sustainability?
What is Company's approach to corporate governance, and how does it impact risk management?
How does Company disclose and communicate risks to investors in its SEC filings?
Management and Corporate Governance:
41. Who are the key executives and board members at Company, and what are their backgrounds?

What is the compensation structure for Company's executive team, including salaries, bonuses, and stock options?
How does Company incentivize and retain key employees?
What is Company's approach to corporate social responsibility and sustainability?
How does Company engage with shareholders and respond to investor concerns?
What are Company's policies regarding conflicts of interest and related-party transactions?
How does Company ensure compliance with relevant laws and regulations, including securities laws?
What is Company's policy on whistleblowing and internal reporting of misconduct?
How does Company foster a culture of innovation and collaboration within the organization?
What are Company's succession planning and talent development strategies?
Legal and Regulatory Compliance:
51. What legal proceedings is Company currently involved in, and what are the potential financial implications?

What are Company's disclosure practices regarding legal risks and contingencies?
How does Company ensure compliance with accounting standards and regulatory requirements?
What regulatory approvals are required for Company's products and operations?
What environmental regulations does Company need to comply with in its manufacturing facilities?
How does Company address intellectual property risks, including patents and trademarks?
What are Company's obligations under labor and employment laws, including worker safety standards?
How does Company disclose information related to government investigations or inquiries?
What are Company's policies for data protection and privacy compliance?
How does Company monitor and address compliance risks in its supply chain?
Capital Structure and Financing:
61. What are Company's sources of funding for capital expenditures and working capital?

How does Company fund its research and development initiatives?
What are Company's financing arrangements for vehicle leasing and customer financing?
How does Company manage its relationships with banks and other financial institutions?
What is Company's dividend policy, if any, and how does it allocate capital for shareholder returns?
How does Company use debt financing, and what are the terms of its outstanding debt?
What are Company's plans for raising additional capital in the future?
How does Company manage its cash and investments to optimize liquidity and returns?
What are Company's policies for managing foreign exchange risks associated with international operations?
How does Company communicate its capital allocation strategy to investors and stakeholders?
Investor Relations and Communication:
71. What information does Company provide to investors in its quarterly earnings releases?

How does Company conduct investor presentations and roadshows?
What channels does Company use to communicate with shareholders and the broader investment community?
How does Company respond to analyst inquiries and investor feedback?
What information does Company disclose in its annual shareholder meetings and proxy statements?
How does Company use social media and other digital platforms for investor relations?
What is Company's policy regarding selective disclosure of material nonpublic information?
How does Company address rumors and speculation in the media that could impact its stock price?
What steps does Company take to maintain transparency and accountability in its communications?
How does Company handle crises and emergencies that may affect its reputation and investor confidence?
Strategic Planning and Growth Initiatives:
81. What are Company's short-term and long-term strategic objectives?

How does Company prioritize growth opportunities in different geographic regions and market segments?
What are Company's expansion plans for its product lineup, including new vehicle models and features?
How does Company approach partnerships and collaborations with other companies?
What is Company's strategy for vertical integration and diversification into new industries?
How does Company evaluate potential mergers and acquisitions as part of its growth strategy?
What are Company's plans for expanding its infrastructure, including charging networks and service centers?
How does Company assess and respond to changes in consumer preferences and market trends?
What role does innovation play in Company's growth strategy, including advancements in technology and design?
How does Company balance growth objectives with risk management and financial sustainability?
Environmental, Social, and Governance (ESG) Factors:
91. What environmental initiatives has Company undertaken to reduce its carbon footprint?

How does Company measure and report its greenhouse gas emissions and energy consumption?
What social responsibility programs does Company support, such as community engagement and philanthropy?
How does Company promote diversity and inclusion within its workforce and supply chain?
What governance practices does Company have in place to ensure accountability and transparency?
How does Company address ethical considerations in its business operations and decision-making?
What steps has Company taken to enhance corporate governance and board oversight?
How does Company engage with stakeholders on ESG issues, including investors, employees, and advocacy groups?
What sustainability goals has Company set, and how does it track progress towards achieving them?
How does Company integrate ESG factors into its overall business strategy and risk management framework?

---


Your financial report should be prescriptive, not just descriptive. Give me an investment conclusion that should be gradually reached. Analyse the document between two *** at the beginning, try to answer the questions between two ---,  and create a real solid report in the format below which is in between two +++ .
+++
Public Equity Analysis
Summarize business, preferably compared across time
● Company description & history
● Product/services
● Customers
● Suppliers
● Competitors and market share
● Risk, regulation, legal
Summarize financial statements, across time
● Financial statements - income statement, balance sheet, cashflow
● Growth metrics
● Profitability metrics
● Financial strength metrics
Assess competitive advantage
● Competitive Rivals
○ The number of competitors
○ Industry growth
○ Similarities in what's offered
○ Exit barriers
○ Fixed costs
● Potential for New Entrants in an Industry
○ Economies of scale
○ Product differentiation
○ Capital requirements
○ Access to distribution channels
○ Regulations
○ Switching costs
● Supplier Power
○ Uniqueness
○ Switching costs
○ Forward integration
○ Industry importance
● Customer Power
○ The number of buyers:
○ Purchase size
○ Switching costs
○ Price sensitivity
○ Informed buyers
 
 ● Threat of Substitutes
○ Relative price performance
○ Customer willingness to go elsewhere
○ The sense that products are similar
○ Availability of close substitutes
Assess management team
● Management background, tenue
● Compensation structure - base, incentives, stock options
● Insider trading (buying/selling)
● Management communication quality
Assess industry trends
● Demographic/socioeconomic trends
● Consumer trends
● Technology trends
● Regulatory trends
● Geopolitical trends
Assess profitability and growth
● Revenue growth drivers
● Cost structure and drivers
● Profitability drivers
Assess valuation
● Project cashflows, perform DCF valuation
● Perform comparables valuation
● Arrive at investment decision
Private Equity Analysis
In addition to public equity analysis, and supplied with proprietary data such as from due diligence process
Assess financial leverage
● Assess fixed income market trends
● Project company cashflows
● Determine deal structure
● Perform sensitivity analysis / stress tests
● Calculate IRR, hurdle rate
● Arrive at investment decision

Venture Capital Analysis
In addition to public equity analysis, and supplied with proprietary data such as from deal pipeline CRM
Assess founding team
● Background, prior success, equity owned
● Intangibles such as grit/motivation, leadership, communication ability
Assess problem & solution
● Describe problem
● Describe solution
● Describe current solutions / substitutes
● Potential size of market
Assess GTM & traction
● Describe strategy
● Describe current traction and results of GTM experiments
Assess market & competition
● Competitor key people, overview, fundraising, product, GTM
Assess fundraising
● Cap table, amount to raise
● Likely return multiples
● Arrive at investment decision
+++

    
    Here's the additional question: {question}
    Helpful Answer:"""
    # TEMPLATE="Short answers only"
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=TEMPLATE,
    )

    def __init__(self) -> None:
        self.model = None
        self.set_pdf_loader()  # to be fixed later
        self.set_textsplitter()  # to be fixed later
        self.set_embedding()  # to be fixed later

    def set_textsplitter(self, text_splitter="RecursiveCharacterTextSplitter"):
        self.text_splitter = text_splitter_factory(text_splitter=text_splitter)
        # all_splits = text_splitter.split_documents(data)

    def set_pdf_loader(self, loader="OnlinePDFLoader"):
        self.loader = pdf_loader_factory(loader=loader)

    def set_embedding(self, embedding="HuggingFaceEmbeddings"):
        self.embedding = HuggingFaceEmbeddings()

    def load_document(self, document_link):
        # document_link = "https://www.apple.com/newsroom/pdfs/FY23_Q1_Consolidated_Financial_Statements.pdf"
        loader = self.loader("uploaded_files/" + document_link)
        data = loader.load()
        all_splits = self.text_splitter.split_documents(data)
        self.vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=self.embedding
        )

    def load_model(self, model_name):  # TO DO : Move model DB to mongo.
        if model_name == "mistral":
            self.model = Mistral7BInstructModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.
        if model_name == "mistral-api":
            print("using api model")
            self.model = MistralAPIModel(
                ""
            ).load_model()  # TO DO : Model configs can be jsonable later on for distribution.

    async def chat(self, query, streaming=True):
        filename = "myfile.txt"
        f = open(filename, "x")
        f.close()

        qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )
        # if streaming:
        async for chunk in qa_chain.astream(
            {"query": query}
        ):  #         async for chunk in qa_chain.astream({"query": query}):
            f = open(filename, "a")
            f.write(chunk["result"] + "\n")
            f.close()

            yield chunk["result"]
        # else:F
        #    return qa_chain({"query": query})['result']
