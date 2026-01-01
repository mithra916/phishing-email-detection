from flask import Flask, render_template, request, jsonify
import joblib
from src.predict import predict_email

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load('models/catboost.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email_text', '')

    if not email_text:
        return jsonify({'error': 'Email text is required'}), 400

    label, confidence = predict_email(email_text, model, vectorizer)

    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 2) if confidence else 'N/A'
    })

if __name__ == '__main__':
    app.run(debug=True)
