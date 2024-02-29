import requests
import streamlit as st
from PIL import Image


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


# Set up layout
st.sidebar.title("Conversations")
# Display conversation list
st.sidebar.text_area("Conversation List", [], height=400, key="conversation_list")

st.title("Chatbot App")

# Upload PDF file
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])


# UI for sending PDF to FastAPI
pdf_button = st.button("Send PDF to FastAPI", key="pdfsend")
if pdf_button and pdf_file:

    response = send_pdf_to_fastapi(pdf_file)
    uploaded_filename = response["filename"]

    st.session_state["uploaded_filename"] = uploaded_filename

    st.write("Response:", response)
elif pdf_button and not pdf_file:
    st.warning("Please upload a PDF file before sending.")


# Chatbot section
st.write("## Chatbot")
input_text = st.text_input("Enter your message:", key="chatbottext")
if st.button("Send"):
    if input_text:
        response = send_query_to_chatbot(
            st.session_state["uploaded_filename"], input_text
        )
        # conversation_list += "\nYou: " + input_text + "\nBot: " + response + "\n"
        st.write(response.content.decode("utf-8"))

    else:
        st.warning("Please enter a message.")
