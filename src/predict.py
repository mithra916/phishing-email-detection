from src.preprocess import clean_email

def predict_email(email_text, model, vectorizer):
    cleaned = clean_email(email_text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vector)[0])
    else:
        confidence = None

    label = "Phishing Email" if prediction == 1 else "Safe Email"
    return label, confidence
