from typing import Optional, Annotated, Union


from fastapi import APIRouter, File, UploadFile

from passlib.context import CryptContext
from core.analyse_engine.analyse import analyser

router = APIRouter()

hash_helper = CryptContext(schemes=["bcrypt"])


@router.post("/data_summary/")
def data_summary():
    return analyser.analyse()
