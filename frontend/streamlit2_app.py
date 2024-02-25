import os
import tempfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_store")
)

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def send_pdf_to_fastapi(pdf_file):
    url = "http://localhost:8080/admin/upload_pdf/"
    files = {"pdf_file": pdf_file}
    response = requests.post(url, files=files)
    return response.json()


# Function to send query to chatbot endpoint
def send_query_to_chatbot(pdf_link, query):
    url = "http://localhost:8080/admin/chat/"
    params = {"pdf_link": pdf_link, "query": query}
    return requests.get(url, params=params, stream=True)


# Define function to interact with backend API
def make_api_request():
    # Code to interact with your backend API goes here
    # Replace this with actual API call
    return "API request made successfully"


def process_documents():

    if not st.session_state.get("pdf_file"):
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            response = send_pdf_to_fastapi(st.session_state["pdf_file"])

            uploaded_filename = response["filename"]
            st.session_state["uploaded_filename"] = uploaded_filename
            st.write("Financial data loaded, you can talk with it!")
        except Exception as e:
            st.error(f"An error occurred: {e}")


def boot():
    #

    if "messages" not in st.session_state:
        st.session_state["messages"] = list()
    st.session_state["pdf_file"] = st.file_uploader(
        label="Upload Documents", type="pdf", accept_multiple_files=False
    )
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    #
    for message in st.session_state.messages:
        print(message[0], message[1])
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)

        response = send_query_to_chatbot(
            st.session_state["uploaded_filename"], query=query
        ).content.decode("utf-8")
        # conversation_list += "\nYou: " + input_text + "\nBot: " + response + "\n"

        st.chat_message("ai").write(response)
        st.session_state.messages.append((query, response))


if __name__ == "__main__":
    #
    boot()
