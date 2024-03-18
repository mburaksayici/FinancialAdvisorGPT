from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apis.v1.routers.admin import router as AdminRouter
from auth.jwt_bearer import JWTBearer
from database.mongo.client import connect

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow requests from any origin (replace "*" with specific origins if needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow these HTTP methods
    allow_headers=["Content-Type"],  # Allow these headers
)

token_listener = JWTBearer()


@app.on_event("startup")
async def start_database():
    connect()


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app."}


# TO DO : Separate that.
app.include_router(AdminRouter, tags=["Administrator"], prefix="/admin")
