from fastapi import Depends, FastAPI

from apis.v1.routers.admin import router as AdminRouter
from apis.v1.routers.student import router as StudentRouter
from auth.jwt_bearer import JWTBearer
from config.config import initiate_database

app = FastAPI()

token_listener = JWTBearer()


@app.on_event("startup")
async def start_database():
    pass
    # await initiate_database()


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic app."}


# TO DO : Separate that.
app.include_router(AdminRouter, tags=["Administrator"], prefix="/admin")
app.include_router(
    StudentRouter,
    tags=["Students"],
    prefix="/student",
    dependencies=[Depends(token_listener)],
)
