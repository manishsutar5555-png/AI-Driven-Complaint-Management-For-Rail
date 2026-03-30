import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

# --------------------------
# 1. Load dataset
# --------------------------
file_path = "train_complaint_dataset.csv"  # change path if needed
df = pd.read_csv(file_path, encoding="latin1")

# --------------------------
# 2. Define urgency levels
# --------------------------
def map_priority(score):
    if score <= 2:
        return "High"
    elif score <= 5:
        return "Medium"
    else:
        return "Low"

df["Urgency"] = df["Sentiment"].apply(map_priority)

# --------------------------
# 3. Train-test split
# --------------------------
X = df["SentimentText"]
y = df["Urgency"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 4. Vectorization + Model
# --------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --------------------------
# 5. Evaluation
# --------------------------
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# --------------------------
# 6. Helper: Clean text
# --------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    return text.strip()

# --------------------------
# 7. Predict priority (updated)
# --------------------------
def predict_priority(text, threshold=0.5):
    """
    Predict the urgency of a complaint based on the trained model.
    Uses probability threshold and input cleaning to avoid bias.
    Returns: 'High', 'Medium', or 'Low'
    """
    text_clean = clean_text(text)
    if len(text_clean) < 3:  # too short / gibberish → Low
        return "Low"
    
    text_vec = vectorizer.transform([text_clean])
    probs = model.predict_proba(text_vec)[0]  # probability per class
    max_prob = max(probs)
    
    if max_prob < threshold:
        return "Low"
    return model.predict(text_vec)[0]

# --------------------------
# 8. Example usage
# --------------------------
sample = ["ac is not working"]
print("Prediction:", predict_priority(sample[0]))
