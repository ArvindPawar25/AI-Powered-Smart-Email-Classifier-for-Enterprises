import joblib

model = joblib.load("email_model.pkl")

print(type(model))
print(model)
