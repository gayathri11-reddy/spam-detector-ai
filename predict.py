import joblib

model = joblib.load("model.pkl")
cv = joblib.load("vectorizer.pkl")

msg = input("Enter message: ")

msg_vector = cv.transform([msg])

print("Result:", model.predict(msg_vector)[0])
