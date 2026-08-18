"""
app.py — ChurnGuard Flask Application

Endpoints:
  GET  /          → Serve the frontend
  POST /predict   → Predict churn probability for a customer
  GET  /metrics   → Return model performance metrics
"""

import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model artifacts
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
with open('metrics.json', 'r') as f:
    metrics = json.load(f)


@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn probability.

    Expects JSON:
    {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000
    }
    """
    try:
        data = request.get_json()

        # Encode categoricals
        gender_encoded = 1 if data['Gender'] == 'Female' else 0
        geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        geo_encoded = geo_map.get(data['Geography'], 0)

        # Build feature array (same order as training)
        features = np.array([[
            float(data['CreditScore']),
            float(geo_encoded),
            float(gender_encoded),
            float(data['Age']),
            float(data['Tenure']),
            float(data['Balance']),
            float(data['NumOfProducts']),
            float(data['HasCrCard']),
            float(data['IsActiveMember']),
            float(data['EstimatedSalary'])
        ]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        probability = float(model.predict_proba(features_scaled)[0][1])
        prediction = int(probability >= 0.5)

        # Determine risk level
        prob_pct = round(probability * 100, 1)
        if prob_pct >= 70:
            risk = 'CRITICAL'
        elif prob_pct >= 50:
            risk = 'HIGH'
        elif prob_pct >= 30:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # Generate explanation
        explanations = []
        if float(data['Age']) > 45:
            explanations.append('Older customers tend to have higher churn rates.')
        if data['Geography'] == 'Germany':
            explanations.append('Customers from Germany show higher churn tendency.')
        if float(data['NumOfProducts']) >= 3:
            explanations.append('Having 3+ products is strongly correlated with churn.')
        if float(data['IsActiveMember']) == 0:
            explanations.append('Inactive members are more likely to churn.')
        if float(data['Balance']) > 100000:
            explanations.append('High account balance can indicate churn risk.')
        if float(data['CreditScore']) < 500:
            explanations.append('Low credit score contributes to churn risk.')
        if not explanations:
            if prob_pct < 30:
                explanations.append('This customer profile shows low churn indicators.')
            else:
                explanations.append('Multiple factors contribute to this prediction.')

        return jsonify({
            'success': True,
            'probability': prob_pct,
            'prediction': prediction,
            'risk': risk,
            'explanations': explanations
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/metrics')
def get_metrics():
    """Return model performance metrics."""
    return jsonify(metrics)


if __name__ == '__main__':
    print("\n  ChurnGuard is running at http://localhost:5050\n")
    app.run(debug=True, port=5050)
