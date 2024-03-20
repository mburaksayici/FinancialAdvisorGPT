from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apis.v1.routers.admin import router as AdminRouter
from auth.jwt_bearer import JWTBearer
from database.mongo.client import connect_mongodb

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the origin of your React application
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],  # Adjust these based on your allowed methods
    allow_headers=["*"],  # Adjust this to the headers your React application sends
)

token_listener = JWTBearer()


@app.on_event("startup")
async def start_database():
    connect_mongodb()


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app."}


# TO DO : Separate that.
app.include_router(AdminRouter, tags=["Administrator"], prefix="/admin")
