import joblib
import pandas as pd

pipeline = joblib.load("models/model.pkl")


def predict(input_dict):
    df = pd.DataFrame([input_dict])
    prediction = pipeline.predict(df)
    return prediction[0]