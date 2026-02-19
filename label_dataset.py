import pandas as pd

# load your merged dataset
df = pd.read_csv( r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's\merged_emails.csv")

# lowercase
df['text'] = df['full_text'].fillna('').str.lower()

# ---------- keyword lists ----------
urgent_kw = [
    'urgent','asap','immediately','deadline','action required','important',
    'priority','right now','without delay','critical','emergency','today',
    'high priority','time sensitive','immediate attention','respond quickly',
    'production down','system down','security breach','fix now'
]

complaint_kw = [
    'not working','issue','error','problem','complaint','failed','failure',
    'broken','unable to','does not work','crashed','stopped working',
    'bad experience','very slow','not responding','incorrect',
    'missing','damaged','delay','late','unsatisfied'
]

request_kw = [
    'please','could you','can you','request','kindly','need help',
    'assist','help me','provide','share','send','give access',
    'schedule','arrange','approve','allow','guide','clarify',
    'explain','let me know'
]
feedback_kw = [
    'feedback','review','suggestion','appreciate','good service','bad service','thank you',
    'great','excellent','well done','satisfied','happy with','not satisfied',
    'improvement','recommend','like the','dislike','experience','rating',
    'comments','opinion'
]

# ---------- labeling ----------
def label_email(row):
    text = row['text']

    # existing spam label
    if str(row['label']).lower() == 'spam' or str(row['label']) == '1':
        return 'spam'

    if any(w in text for w in urgent_kw):
        return 'urgent'

    if any(w in text for w in complaint_kw):
        return 'complaint'

    if any(w in text for w in request_kw):
        return 'request'

    if any(w in text for w in feedback_kw):
        return 'feedback'

    return 'others'

df['final_label'] = df.apply(label_email, axis=1)

# -------- BALANCING --------
from sklearn.utils import resample
import pandas as pd
from sklearn.utils import resample

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"
input_path = base + r"\cleaned_emails.csv"
output_path = base + r"\final_balanced_dataset.csv"

df = pd.read_csv(input_path, encoding="latin1")

# ---------- LABEL MAPPING ----------
def map_label(label):
    label = str(label).lower().strip()

    if label == "spam":
        return "spam"

    elif label == "complaint":
        return "complaint"

    elif label == "request":
        return "request"

    elif label == "support":
        return "request"

    elif label == "incident":
        return "urgent"

    elif label == "feedback":
        return "feedback"

    else:
        return "others"

df['final_label'] = df['label'].apply(map_label)

# keep required columns
df = df[['full_text','final_label']]

print("\nBefore balancing:")
print(df['final_label'].value_counts())

# ---------- BALANCING ----------
max_size = df['final_label'].value_counts().max()

balanced_df = pd.concat([
    resample(group,
             replace=True,
             n_samples=max_size,
             random_state=42)
    for label, group in df.groupby('final_label')
])

# shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42)

balanced_df.to_csv(output_path, index=False)

print("\nBalanced dataset created!")
print(balanced_df['final_label'].value_counts())

max_size = df['final_label'].value_counts().max()

balanced_df = pd.concat([
    resample(group,
             replace=True,
             n_samples=max_size,
             random_state=42)
    for label, group in df.groupby('final_label')
])

balanced_df = balanced_df.sample(frac=1, random_state=42)  # shuffle

balanced_df[['full_text','final_label']].to_csv("balanced_email_dataset.csv", index=False)

print("\nBalanced Dataset:")
print(balanced_df['final_label'].value_counts())

