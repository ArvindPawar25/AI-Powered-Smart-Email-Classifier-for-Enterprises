import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
data_path = base + r"\english_only_emails.csv"

# ---------- LOAD DATA ----------
df = pd.read_csv(data_path, encoding="latin1")
df['full_text'] = df['full_text'].fillna("")

X_text = df['full_text']
y = df['label']

# ---------- TF-IDF WITH BIGRAMS ----------
vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1,2)   # unigram + bigram
)

X = vectorizer.fit_transform(X_text)

print("Feature shape:", X.shape)
print("Classes:", y.unique())

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------- TRAIN MODEL ----------
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# ---------- PREDICT ----------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------- SAVE ----------
joblib.dump(model, "email_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
