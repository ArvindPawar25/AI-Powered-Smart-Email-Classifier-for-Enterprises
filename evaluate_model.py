import pandas as pd
import joblib
import random

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
data_path = base + r"\english_only_emails.csv"

# ---------- LOAD ----------
df = pd.read_csv(data_path, encoding="latin1")
df['full_text'] = df['full_text'].fillna("")

model = joblib.load("email_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------- TRANSFORM ----------
X = vectorizer.transform(df['full_text'])
y_true = df['label']

# ---------- PREDICT ----------
y_pred = model.predict(X)

# ---------- METRICS ----------
print("\n=========== FINAL MODEL PERFORMANCE ===========")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ---------- SAMPLE PREDICTIONS ----------
print("\n=========== SAMPLE EMAIL TEST ===========\n")

for i in random.sample(range(len(df)), 5):
    print("EMAIL:")
    print(df.iloc[i]['full_text'][:250])
    print("ACTUAL:", df.iloc[i]['label'])
    print("PREDICTED:", y_pred[i])
    print("-"*60)
