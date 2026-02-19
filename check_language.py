import pandas as pd
import re

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
path = base + r"\cleaned_emails.csv"

df = pd.read_csv(path, encoding="latin1")

def is_english(text):
    text = str(text)

    # if contains non-ascii characters → probably non-English
    if re.search(r'[^\x00-\x7F]', text):
        return False

    # if too many non-letters
    letters = sum(c.isalpha() for c in text)
    ratio = letters / (len(text) + 1)

    return ratio > 0.6

df['english'] = df['full_text'].apply(is_english)

print("\nTotal emails:", len(df))
print("English emails:", df['english'].sum())
print("Non-English emails:", len(df) - df['english'].sum())

print("\nSample Non-English Emails:\n")
print(df[df['english'] == False]['full_text'].head(10))
