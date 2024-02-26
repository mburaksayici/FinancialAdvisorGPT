from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from passlib.context import CryptContext

from auth.jwt_handler import sign_jwt
from core.engine.data_pipelines.news.template import NewsDataChain
from core.engine.data_pipelines.stocks.template import StockDataChain

# from core.engine.model_driver import ModelDriver
from core.engine.online_model_driver import OnlineModelDriver
from database.database import add_admin
from models.admin import Admin
from schemas.admin import AdminData, AdminSignIn, ChatRequest

model_driver = OnlineModelDriver()
model_driver.set_pdf_loader("PyPDFLoader")
model_driver.load_model("mistral-api")
model_driver.set_data_chains([NewsDataChain, StockDataChain])
router = APIRouter()

hash_helper = CryptContext(schemes=["bcrypt"])


@router.post("/login")
async def admin_login(admin_credentials: AdminSignIn = Body(...)):
    admin_exists = await Admin.find_one(Admin.email == admin_credentials.username)
    if admin_exists:
        password = hash_helper.verify(admin_credentials.password, admin_exists.password)
        if password:
            return sign_jwt(admin_credentials.username)

        raise HTTPException(status_code=403, detail="Incorrect email or password")

    raise HTTPException(status_code=403, detail="Incorrect email or password")


@router.post("", response_model=AdminData)
async def admin_signup(admin: Admin = Body(...)):
    admin_exists = await Admin.find_one(Admin.email == admin.email)
    if admin_exists:
        raise HTTPException(
            status_code=409, detail="Admin with email supplied already exists"
        )

    admin.password = hash_helper.encrypt(admin.password)
    new_admin = await add_admin(admin)
    return new_admin


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
def chat(pdf_link: str, query: str):
    print("CHATTING")
    model_driver.load_document(pdf_link)

    return model_driver.chat(query)


# return model_driver.chat(query)


@router.post("/upload_pdf/")
def upload_pdf(pdf_file: UploadFile = File(...)):
    with open(f"uploaded_files/{pdf_file.filename}", "wb") as buffer:
        buffer.write(pdf_file.file.read())
    return {"filename": pdf_file.filename}
