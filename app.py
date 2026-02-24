from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# =========================
# SAFE MODEL LOADING
# =========================

model = None
vectorizer = None

try:
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print("Model loading failed:", e)

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.lower()

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "")

    if not message.strip():
        return render_template("index.html", prediction="Please enter a message!")

    if model and vectorizer:
        cleaned = clean_text(message)
        data = vectorizer.transform([cleaned])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0].max()
        confidence = round(probability * 100, 2)
    else:
        # fallback rule-based detection
        spam_words = ["win", "free", "offer", "money", "lottery", "urgent", "click"]
        prediction = 1 if any(word in message.lower() for word in spam_words) else 0
        confidence = 80 if prediction == 1 else 90

    if prediction == 1:
        result = "Spam ❌"
    else:
        result = "Not Spam ✅"

    return render_template("index.html", prediction=result, confidence=confidence)

# =========================
# RUN SERVER (RENDER SAFE)
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)