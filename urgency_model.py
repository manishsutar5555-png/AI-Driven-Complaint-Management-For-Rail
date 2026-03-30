import joblib

# Load saved model + vectorizer
urgency_model = joblib.load("complaint_priority_model.pkl")
vectorizer = joblib.load("complaint_vectorizer.pkl")

def predict_urgency(text: str) -> str:
    vec = vectorizer.transform([text])
    return urgency_model.predict(vec)[0]
