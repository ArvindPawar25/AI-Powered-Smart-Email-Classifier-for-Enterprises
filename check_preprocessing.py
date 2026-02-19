import pandas as pd

path = r"C:\Users\Arvind Pawar\Desktop\Btech\Infosys SpringBoard\Datasets\Mam's\cleaned_emails.csv"

df = pd.read_csv(path, encoding="latin1")

print("\n========= DATASET INFO =========")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# empty rows
empty_rows = df['full_text'].isna().sum()
print("\nEmpty text rows:", empty_rows)

# short rows
short_rows = (df['full_text'].str.len() < 10).sum()
print("Very short text rows:", short_rows)

# label distribution
print("\n========= LABEL DISTRIBUTION =========")
print(df['label'].value_counts())

# sample data
print("\n========= SAMPLE EMAILS =========")
print(df[['label','full_text']].sample(5, random_state=42))

# longest email
print("\n========= LONGEST EMAIL =========")
print(df.iloc[df['full_text'].str.len().idxmax()]['full_text'][:500])

print("\n========= SHORTEST EMAIL =========")
print(df.iloc[df['full_text'].str.len().idxmin()]['full_text'])
