import json

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

model = CatBoostClassifier()
model.load_model('titanic_model.cbm')

with open('model_columns.json') as f:
    model_columns = json.load(f)


class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: bool
    Embarked_S: bool

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.__dict__])
    data = data[model_columns]
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "survived:": bool(prediction),
        "probability": float(probability)
    }