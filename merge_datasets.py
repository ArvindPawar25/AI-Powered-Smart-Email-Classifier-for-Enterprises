import pandas as pd

base = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's"

emails_path = base + r"\Emails_data.csv"
spam_path = base + r"\Spam_ham_emails.csv"
extra_path = base + r"\merged_emails2.csv"
output_path = base + r"\merged_emails.csv"

# ---------------- LOAD ----------------
emails_df = pd.read_csv(emails_path, encoding="latin1")
spam_df = pd.read_csv(spam_path, encoding="latin1")
extra_df = pd.read_csv(extra_path, encoding="latin1")

# ---------------- DATASET 1 ----------------
emails_df['full_text'] = emails_df['subject'].fillna('') + " " + emails_df['body'].fillna('')
emails_df = emails_df[['full_text','type']]
emails_df.rename(columns={'type':'label'}, inplace=True)

# ---------------- DATASET 2 ----------------
spam_df = spam_df[['Text','Spam']]
spam_df.rename(columns={'Text':'full_text','Spam':'label'}, inplace=True)
spam_df['label'] = spam_df['label'].map({1:'spam',0:'ham'})

# ---------------- DATASET 3 ----------------
extra_df = extra_df[['text','label']]
extra_df.rename(columns={'text':'full_text'}, inplace=True)

# ---------------- MERGE ----------------
merged_df = pd.concat([emails_df, spam_df, extra_df], ignore_index=True)

# lowercase labels
merged_df['label'] = merged_df['label'].astype(str).str.lower()

# ❌ REMOVE unwanted classes
remove_labels = ['change','problem','ham']
merged_df = merged_df[~merged_df['label'].isin(remove_labels)]

# clean
merged_df.dropna(subset=['full_text'], inplace=True)
merged_df.drop_duplicates(subset=['full_text'], inplace=True)

merged_df.to_csv(output_path, index=False)

print("Filtered dataset created!")
print(merged_df['label'].value_counts())
