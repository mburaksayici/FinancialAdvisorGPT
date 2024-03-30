from typing import Optional, Annotated, Union


from fastapi import APIRouter, Body, File, HTTPException, UploadFile, Header
from fastapi.responses import StreamingResponse
from passlib.context import CryptContext

from auth.jwt_handler import sign_jwt
from core.engine.data_pipelines.news.template import NewsDataChain
from core.engine.data_pipelines.stocks.template import StockDataChain
from core.engine.data_pipelines.db_retrieval.template import DBRetrievalChain

# from core.engine.model_driver import ModelDriver
from core.engine.online_model_driver import OnlineModelDriver
from models.users.users import Users
from schemas.admin import AdminData, AdminSignIn, ChatRequest

model_driver = OnlineModelDriver()
model_driver.set_pdf_loader("PyPDFLoader")
model_driver.load_model("mistral-api")
model_driver.set_data_chains(
    [DBRetrievalChain, NewsDataChain, StockDataChain]
)  # NewsDataChain StockDataChain
model_driver.load_assets()
router = APIRouter()

hash_helper = CryptContext(schemes=["bcrypt"])

"""
@router.post("/login")
async def admin_login(admin_credentials: AdminSignIn = Body(...)):
    admin_exists = await Admin.find_one(Admin.email == admin_credentials.username)
    if admin_exists:
        password = hash_helper.verify(admin_credentials.password, admin_exists.password)
        if password:
            return sign_jwt(admin_credentials.username)

        raise HTTPException(status_code=403, detail="Incorrect email or password")

    raise HTTPException(status_code=403, detail="Incorrect email or password")
"""


@router.post(
    "/add_user",
)
def add_user(user_name, user_password):
    users_doc = Users(user_name=user_name, user_password=user_password)
    users_doc.save()
    return user_name


"""
@router.get("/chat/")
async def chat(pdf_link: str, query: str):
    print("CHATTING")
    model_driver.load_document(pdf_link)

    async def generate():
        async for chunk in model_driver.chat(query):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")

    # return StreamingResponse(model_driver.chat(query), media_type='text/event-stream')

"""


@router.get("/chat/")
def answer(
    query: str,
    pdf_link: str = None,
):
    print("CHATTING")
    if pdf_link:
        model_driver.load_document(pdf_link)

    return {"answer": model_driver.answer(query)}


@router.get("/initialise_conversation/")
def initialise_conversation(
    user_id: Annotated[Union[str, None], Header()] = "admin",
):
    return {"conversation_id": model_driver.initialise_conversation(user_id=user_id)}


@router.get("/conversation/")
def conversation(query: str, conversation_id: str, user_id: str):
    print("conversation...")  #  TO DO : use python logging later.
    return model_driver.conversation(
        query=query, user_id=user_id, conversation_id=conversation_id
    )


@router.post("/upload_pdf/")
def upload_pdf(pdf_file: UploadFile = File(...)):
    with open(f"uploaded_files/{pdf_file.filename}", "wb") as buffer:
        buffer.write(pdf_file.file.read())
    return {"filename": pdf_file.filename}
