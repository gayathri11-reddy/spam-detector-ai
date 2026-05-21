import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = {
    "Message": [
        "Win money now",
        "Claim your free prize",
        "Hello friend how are you",
        "Let's meet tomorrow",
        "You won cash prize",
        "Free entry in contest",
        "Are you coming today",
        "Call me later"
    ],
    "Label": [
        "Spam","Spam","Not Spam","Not Spam",
        "Spam","Spam","Not Spam","Not Spam"
    ]
}

df = pd.DataFrame(data)

cv = CountVectorizer()
X = cv.fit_transform(df["Message"])
y = df["Label"]

model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(cv, "vectorizer.pkl")

print("Model trained successfully")
