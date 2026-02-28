import joblib

MODEL_PATH = r"C:\Users\desal\spam-detector-ml\src\spam_model.joblib"
model = joblib.load(MODEL_PATH)

def predict_spam(text: str):
    pred = model.predict([text])[0]
    return "spam" if pred == 1 else "not_spam"

if __name__ == "__main__":
    sample = "Congratulations! You've won iPhone. Click here to claim your prize."
    print("Prediction:", predict_spam(sample))