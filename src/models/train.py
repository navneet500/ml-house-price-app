import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]
TARGET = "SalePrice"


def train():
    df = pd.read_csv("data/raw/train.csv")

    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ PIPELINE (core improvement)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # ✅ Evaluation
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"✅ RMSE: {rmse}")
    print(f"✅ R2 Score: {r2}")

    # ✅ Save FULL pipeline (not just model)
    joblib.dump(pipeline, "models/model.pkl")

    print("🚀 Pipeline model saved!")


if __name__ == "__main__":
    train()