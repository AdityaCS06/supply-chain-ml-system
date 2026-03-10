from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/demand_model.pkl")

class DemandInput(BaseModel):
    price: float
    availability: int
    stock_levels: int
    lead_times: int


@app.post("/predict-demand")
def predict(data: DemandInput):

    features = [[
        data.price,
        data.availability,
        data.stock_levels,
        data.lead_times
    ]]

    prediction = model.predict(features)

    return {"prediction": prediction.tolist()}