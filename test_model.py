# test_model.py
from src.predict import predict_email
import joblib

# Load saved TF-IDF vectorizer and CatBoost model
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/catboost.pkl')  # pick the best model

# Test with a sample email
test_email = "Urgent! Verify your account immediately."
label, confidence = predict_email(test_email, model, vectorizer)

print(f"Prediction: {label}")
print(f"Confidence: {confidence if confidence else 'N/A'}")
