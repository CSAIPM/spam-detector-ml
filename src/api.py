from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Detector API")

MODEL_PATH = r"C:\Users\desal\spam-detector-ml\src\spam_model.joblib"
model = joblib.load(MODEL_PATH)

class MessageInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: MessageInput):
    pred = model.predict([input.text])[0]
    label = "spam" if pred == 1 else "not_spam"
    return {"label": label}