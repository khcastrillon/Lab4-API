from __future__ import annotations
from typing import List
from pydantic import BaseModel
from DataModel import DataModel

class DataList(BaseModel):
    __root__: List[DataModel]

class Config:
    orm_mode = True

