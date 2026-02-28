import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load dataset
file_path = r"C:\Users\desal\spam-detector-ml\data\spam.csv"
# Use latin-1 encoding as it's common for this specific dataset
df = pd.read_csv(file_path, encoding="latin-1")

# 2. Fix Columns
# Select only the first two columns (label and message) regardless of their names
df = df.iloc[:, [0, 1]] 
df.columns = ["label", "message"]

# 3. Clean and Normalize Data
# Force messages to string to avoid AttributeError during TF-IDF
df["message"] = df["message"].astype(str) 

# Normalize labels: remove whitespace, lowercase, and map to numbers
# This prevents the 'n_samples=0' error by ensuring the map actually finds matches
df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})

# 4. Remove empty or failed rows
df = df.dropna(subset=["label"])

# --- SAFETY CHECK ---
print(f"Total processed rows: {len(df)}")
if len(df) == 0:
    print("ERROR: Dataset is empty after cleaning!")
    print("Actual values found in your label column were:", pd.read_csv(file_path, encoding="latin-1").iloc[:, 0].unique())
    exit()
# --------------------

# 5. Define Features and Target
X = df["message"]
y = df["label"]

# 6. Train-test split
# 'stratify=y' ensures the spam/ham ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 7. Build the Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# 8. Train the Model
print("Training model...")
pipeline.fit(X_train, y_train)

# 9. Evaluate
print("\nModel Evaluation Report:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Save the Model
save_path = r"C:\Users\desal\spam-detector-ml\src\spam_model.joblib"
joblib.dump(pipeline, save_path)
print(f"\nSuccess! Model saved to {save_path}")

# 11. Live Test
print("\n--- Live Test ---")
test_samples = ["WINNER! Claim your $500 cash prize now.", "Hey, are we still meeting for coffee?"]
predictions = pipeline.predict(test_samples)
for text, pred in zip(test_samples, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}]: {text}")