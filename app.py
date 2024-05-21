from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import secrets
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials



from apis.v1.routers.admin import router as AdminRouter
from apis.v1.routers.dashboard_page import router as DashboardRouter

from auth.jwt_bearer import JWTBearer
from database.mongo.client import connect_mongodb

app = FastAPI(docs_url=None, redoc_url=None, openapi_url = None)

security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "yabu")
    correct_password = secrets.compare_digest(credentials.password, "yabu")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


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

@app.get("/docs")
async def get_documentation(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json")
async def openapi(username: str = Depends(get_current_username)):
    return get_openapi(title = "FastAPI", version="0.1.0", routes=app.routes)



# TO DO : Separate that.
app.include_router(AdminRouter, tags=["Administrator"], prefix="/admin")
