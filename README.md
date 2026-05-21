# 📧 Spam Detector AI

An end-to-end Machine Learning project that detects whether a message is **Spam or Not Spam** using Natural Language Processing (NLP) and the **Multinomial Naive Bayes algorithm**.

---

## 🚀 Project Overview

This project demonstrates how machine learning can be used to automatically classify text messages. It processes raw text data, converts it into numerical features, and applies a classification model to detect spam messages.

---

## 🧠 Features

- Classifies messages as Spam or Not Spam
- Uses NLP techniques for text processing
- Converts text using CountVectorizer
- Machine Learning model: Multinomial Naive Bayes
- Saves trained model for reuse
- Simple and beginner-friendly implementation

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas 📊
- Scikit-learn 🤖
- NLP (CountVectorizer)

---

## 📁 Project Structure
spam-detector-ai/
│
├── train_model.py        # Trains the Naive Bayes model and saves it
├── predict.py            # Loads model and predicts new messages
├── model.pkl             # Saved trained ML model (generated after training)
├── vectorizer.pkl        # Saved CountVectorizer (generated after training)
├── requirements.txt      # Required Python libraries
├── dataset.csv           # Sample dataset (optional but recommended)
└── README.md             # Project documentation
