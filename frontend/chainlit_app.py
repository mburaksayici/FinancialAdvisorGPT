import io
import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import chainlit as cl

from core.engine.driver import ModelDriver

model_driver = ModelDriver()
model_driver.set_pdf_loader("PyPDFLoader")
model_driver.load_model("mistral-api")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("burak", "sean"):  # temporary solution
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path
    await cl.Message(
        content="Hello there, Welcome to AskAnyQuery related to Data!"
    ).send()

    files = None

    # Waits for user to upload csv data
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
        ).send()

    # load the csv data and store in user_session
    file = files[0]
    # creating user session to store data

    cl.user_session.set("data", file.path)
    model_driver.load_document(file.path)
    cl.user_session.set("chain", model_driver.get_chain())

    # Send response back to user
    await cl.Message(
        content=f"`{file.name}` uploaded! Now you ask me anything related to your data"
    ).send()


@cl.on_message
async def main(message: str):

    """cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
    )
    """
    msg = cl.Message(content="")

    await msg.send()

    # Send a response back to the user
    # async for chunk in model_driver.chat_chain_injected(message,qa_chain=cl.user_session.get('chain')):
    #    await msg.stream_token(chunk)

    chain = cl.user_session.get("chain")
    async for chunk in chain.astream({"query": message.content}):
        await msg.stream_token(str(chunk["result"]))

    await msg.update()
