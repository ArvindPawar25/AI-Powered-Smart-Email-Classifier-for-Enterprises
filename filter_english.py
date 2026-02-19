import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # consistent results

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
path = base + r"\cleaned_emails.csv"
out = base + r"\english_only_emails.csv"

df = pd.read_csv(path, encoding="latin1")

def keep_english(text):
    try:
        text = str(text)[:200]   # only check first 200 chars (FAST)
        return detect(text) == "en"
    except:
        return False

df['is_english'] = df['full_text'].apply(keep_english)

filtered = df[df['is_english']].drop(columns=['is_english'])
filtered.to_csv(out, index=False)

print("Original rows:", len(df))
print("English rows:", len(filtered))
