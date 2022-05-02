from typing import Optional
import pandas as pd
# from joblib import load
from fastapi import FastAPI
from DataModel import DataModel
import PredictionModel

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    X = df.drop('Life expectancy', axis=1)
    y = df['Life expectancy']
    # model = load("assets/pipeline.joblib")
    result = PredictionModel.Model.make_predictions(X)
    # result = model.predict(df)
    return result

@app.post("/predictR2")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    X = df.drop('Life expectancy', axis=1)
    y = df['Life expectancy']
    # model = load("assets/pipeline.joblib")
    result = PredictionModel.Model.R2(X, y)
    # result = model.predict(df)
    return result