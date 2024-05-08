from langchain.text_splitter import RecursiveCharacterTextSplitter
import textdistance as td

# TO DO : Need baseclass for textsplitters.


def text_splitter_factory(text_splitter="RecursiveCharacterTextSplitter"):
    if text_splitter == "RecursiveCharacterTextSplitter":
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)


def text_similarity_finder(word, word_list, algorithm="cosine"):
    results = list()
    if algorithm == "cosine":
        for i, compare_word in enumerate(word_list):
            score = td.sorensen(word.lower(), compare_word.lower())
            if score > 0.4:
                results.append({"score": score, "word": compare_word})

    sorted_list = sorted(results, key=lambda d: d["score"], reverse=True)
    return sorted_list
