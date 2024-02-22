from langchain.text_splitter import RecursiveCharacterTextSplitter

# TO DO : Need baseclass for textsplitters.


def text_splitter_factory(text_splitter="RecursiveCharacterTextSplitter"):
    if text_splitter == "RecursiveCharacterTextSplitter":
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
