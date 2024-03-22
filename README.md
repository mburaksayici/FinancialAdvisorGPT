RAG Pipeline with LLM 



<img width="497" alt="Ekran Resmi 2024-03-23 01 18 00" src="https://github.com/mburaksayici/finsean/assets/25187211/81666cc3-c988-45ce-9e58-68c3dc39d452">


Database Choice:

Although i dont utilise it atm, database choice is mongodb latest version with its Vector DB feature.
Although I use Chromadb for now, I don't really believe the options like chromadb(which is sqlite) is necessary, big DB companies will offer faster Vector DB features.

I also have done Vector similarity search in Gesund Inc., its just keeping arrays and retrieving it.

Cache Choice:

Although I haven't activated it as of the date of 22 March, 2024, Redis to cache conversations between user and LLM architecture is extremely needed.

Parallelization of Data Pipelines:

RAG takes time.  Data acquisition, augmentation wrt the new data, feeding LLMs with new data takes time. Thus, each independent pipeline is parallelised. 

The RAG pipeline diagram below, DataPipelines are responsible for both generating questions to search for. DBRetrieval creates 10 questions given user prompt, then do vector search on DB which extracted from document. StockDataRetrieval creates 10 API request payloads given user prompt oversimplified like {"tinker":"TSL", "data": "MARKETCAP"}, then do API calls to get live data. Search of 10 questions, all parallelised.  It takes 45 secs to analyse 20 pdfs (30 pages on average), stock data of 2 companies and 10 news with parallelisation. Otherwise, it takes 3-4 mins to create example report above.



RAG Pipeline: 

![Baslksz_Diyagram drawio-2](https://github.com/mburaksayici/finsean/assets/25187211/07ab079d-3da0-41f6-8796-0fcf884b6d7e)


LLM Model:

Mistral-Tiny and Mistral-Small is used. Mistral-Tiny not always following the instructions, small is better.
Big/better models are also error-prone. Chip Huyen's blog has great insights and production experiences that I'll summarize, but the most interesting one is that :
"A couple of people who’ve worked with LLMs for years told me that they just accepted this ambiguity and built their workflows around that. It’s a different mindset compared to developing deterministic programs, but not something impossible to get used to."
So it means, sometimes you can do multiple requests to get what you need from LLM.  One case is that in StockDataChain request, I request python dict directly from LLM, and I do eval(str) which LLM sometimes doesn't produce in python-syntax. Retry again, it'll work. 


Multiple models can be used, for small tasks, cheap LLMs can be used. Or, some local small LLM models are also OK to use. Or, first  trial with small model and second trial with expensive models are also fine.

![Ekran_Resmi_2024-02-23_12 09 22](https://github.com/mburaksayici/finsean/assets/25187211/ac98014f-cfdb-4692-bdeb-fd642341d328)











Some insights(from https://huyenchip.com/2023/04/11/llm-engineering.html):


1. Ambiguous output format : there’s no guarantee that the outputs will always follow this format .
2. You can force an LLM to give the same response by setting temperature = 0, which is, in general, a good practice. While it mostly solves the consistency problem, it doesn’t inspire trust in the system.
3.  A couple of people who’ve worked with LLMs for years told me that they just accepted this ambiguity and built their workflows around that. It’s a different mindset compared to developing deterministic programs, but not something impossible to get used to. !!!
4. Prompt versioning : i plan to do that as well.
5. Prompt optimization : A research area.
6. Input tokens can be processed in parallel, which means that input length shouldn’t affect the latency that much. This principle is applied.
7. "The impossibility of cost + latency analysis for LLMs : The LLM application world is moving so fast that any cost + latency analysis is bound to go outdated quickly. Matt Ross, a senior manager of applied research at Scribd, told me that the estimated API cost for his use cases has gone down two orders of magnitude over the last year. Latency has significantly decreased as well."
8. "Prompt tuning- finetuning. : increase the number of examples, finetuning will give better model performance than prompting." It's same for other ml models but nice to read this exp.
9. Finetuning with distillation : a research area.
10. "If 2021 was the year of graph databases, 2023 is the year of vector databases." we 'll see.. Old-school DBs are also offering vector db feature since its just a list of numbers(vectors in DNNs).
11. One argument I often hear is that prompt rewriting shouldn’t be a problem because:
"Newer models should only work better than existing models." I’m not convinced about this. Newer models might, overall, be better, but there will be use cases for which newer models are worse.
12. "Experiments with prompts are fast and cheap, as we discussed in the section Cost." : While I agree with this argument, a big challenge I see in MLOps today is that there’s a lack of centralized knowledge for model logic, feature logic, prompts, etc. 
14. Control flows: sequential, parallel, if, for loop:  Shared the image above.  
15. "For example, if you want your agent to choose between three actions search, SQL executor, and Chat, you might explain how it should choose one of these actions as follows (very approximate), In other words, you can use LLMs to decide the condition of the control flow! " . Currently used in this project. Like :  "IF DO WE NEED STOCK DATA FOR A QUESTION, LLM DECIDES THAT IN THE CODEBASE!"


