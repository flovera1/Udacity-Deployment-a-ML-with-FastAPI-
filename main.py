from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "welcome to the ML API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Align columns with training
    model_columns = model.feature_names_in_
    df = df.reindex(columns=model_columns, fill_value=0)

    preds = model.predict(df)
    return {"prediction": int(preds[0])}