from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.models.predict import predict

app = FastAPI()


class HouseInput(BaseModel):
    OverallQual: int = Field(..., example=7)
    GrLivArea: int = Field(..., example=1500)
    GarageCars: int = Field(..., example=2)
    TotalBsmtSF: int = Field(..., example=800)


@app.get("/")
def home():
    return {"message": "ML Pipeline API 🚀"}


@app.post("/predict")
def get_prediction(data: HouseInput):
    result = predict(data.dict())
    return {"predicted_price": result}