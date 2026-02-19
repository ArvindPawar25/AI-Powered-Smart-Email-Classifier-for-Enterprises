import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
input_path = base + r"\merged_emails.csv"
output_path = base + r"\cleaned_emails.csv"

df = pd.read_csv(input_path, encoding="latin1")

# ---------- CLEAN FUNCTION ----------
def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    text = " ".join(words)

    return text

# apply cleaning
df['full_text'] = df['full_text'].apply(clean_text)

# remove empty rows
df = df[df['full_text'].str.len() > 5]

df.to_csv(output_path, index=False)

print("Text cleaning + stopwords removal completed!")
print("Total rows:", len(df))
