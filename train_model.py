"""
train_model.py — Train XGBoost churn model and save artifacts.

Generates a synthetic dataset matching the Churn_Modelling.csv schema,
trains multiple classifiers for comparison, and saves:
  - model.pkl      (trained XGBoost classifier)
  - scaler.pkl     (fitted StandardScaler)
  - metrics.json   (accuracy, precision, recall, f1, feature importances, classifier comparison)
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def generate_synthetic_data(n_samples=10000, random_state=42):
    """Generate synthetic churn dataset matching Kaggle schema."""
    rng = np.random.RandomState(random_state)

    credit_score = rng.normal(650, 96, n_samples).clip(350, 850).astype(int)
    age = rng.normal(39, 10, n_samples).clip(18, 92).astype(int)

    geography_choices = ['France', 'Germany', 'Spain']
    geography_probs = [0.5014, 0.2509, 0.2477]
    geography = rng.choice(geography_choices, n_samples, p=geography_probs)

    gender_choices = ['Male', 'Female']
    gender_probs = [0.5457, 0.4543]
    gender = rng.choice(gender_choices, n_samples, p=gender_probs)

    tenure = rng.randint(0, 11, n_samples)
    balance = np.where(
        rng.random(n_samples) < 0.36,
        0.0,
        rng.normal(76486, 62397, n_samples).clip(0, 250000)
    )
    num_of_products = rng.choice([1, 2, 3, 4], n_samples, p=[0.5084, 0.4590, 0.0266, 0.0060])
    has_cr_card = rng.choice([0, 1], n_samples, p=[0.2945, 0.7055])
    is_active_member = rng.choice([0, 1], n_samples, p=[0.4849, 0.5151])
    estimated_salary = rng.uniform(11, 199992, n_samples)

    # Generate target with realistic churn rate (~20%) and STRONG learnable signals
    # Using a logistic-style approach so the model can learn clear patterns
    churn_logit = np.zeros(n_samples, dtype=float)

    # Age: strong positive effect (older = more churn)
    churn_logit += (age - 38) * 0.06

    # Credit Score: strong negative effect (lower score = more churn)
    churn_logit += (650 - credit_score) * 0.008

    # Geography: Germany has much higher churn
    churn_logit += (geography == 'Germany') * 1.5
    churn_logit += (geography == 'Spain') * -0.2

    # Gender: Female slightly higher churn
    churn_logit += (gender == 'Female') * 0.5

    # Number of Products: 3+ products = very high churn
    churn_logit += (num_of_products == 1) * -0.2
    churn_logit += (num_of_products >= 3) * 2.5
    churn_logit += (num_of_products >= 4) * 2.0

    # Active member: inactive = higher churn
    churn_logit += (is_active_member == 0) * 0.8

    # Balance: high balance = more churn (counterintuitive but matches real data)
    churn_logit += (balance > 100000) * 0.5
    churn_logit += (balance > 150000) * 0.3
    churn_logit += (balance == 0) * -0.4

    # Tenure: long tenure = less churn
    churn_logit += (tenure < 2) * 0.3
    churn_logit += (tenure > 7) * -0.3

    # Has credit card: slight effect
    churn_logit += (has_cr_card == 0) * 0.15

    # Salary: lower salary = slightly more churn
    churn_logit += (estimated_salary < 50000) * 0.2

    # Base intercept to target ~20% churn
    churn_logit += -2.0

    # Convert logit to probability via sigmoid + noise
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
    noise = rng.normal(0, 0.05, n_samples)
    churn_prob = (churn_prob + noise).clip(0.01, 0.99)
    exited = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Exited': exited
    })

    return df


def preprocess(df):
    """Encode categoricals and split features/target."""
    df = df.copy()
    df['Gender'] = (df['Gender'] == 'Female').astype(int)

    geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    df['Geography'] = df['Geography'].map(geo_map)

    X = df.drop('Exited', axis=1)
    y = df['Exited']
    return X, y


def main():
    print("=" * 60)
    print("  ChurnGuard — Model Training Pipeline")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/5] Generating synthetic dataset (10,000 samples)...")
    df = generate_synthetic_data()
    print(f"  -> Dataset shape: {df.shape}")
    print(f"  -> Churn rate: {df['Exited'].mean():.1%}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing data...")
    X, y = preprocess(df)

    # 3. Scale features
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    feature_names = list(X.columns)
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 4. Train multiple classifiers
    print("\n[4/5] Training classifiers...")
    names = [
        "Logistic Regression", "Nearest Neighbors", "Naive Bayes",
        "Decision Tree", "Random Forest", "AdaBoost",
        "Gradient Boosting", "Neural Net", "XGBoost"
    ]
    classifiers = [
        LogisticRegression(max_iter=1000),
        KNeighborsClassifier(5),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=100, random_state=0),
        AdaBoostClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
        MLPClassifier(alpha=1, max_iter=1000, random_state=0),
        xgb.XGBClassifier(eval_metric='logloss', random_state=0)
    ]

    classifier_scores = {}
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = round(clf.score(X_test, y_test) * 100, 1)
        classifier_scores[name] = score
        print(f"  {name:25s} -> {score}%")

    # 5. Final XGBoost model + metrics
    print("\n[5/5] Generating final metrics with XGBoost...")
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=0)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
    precision = round(precision_score(y_test, y_pred) * 100, 1)
    recall = round(recall_score(y_test, y_pred) * 100, 1)
    f1 = round(f1_score(y_test, y_pred) * 100, 1)

    print(f"\n  Accuracy:  {accuracy}%")
    print(f"  Precision: {precision}%")
    print(f"  Recall:    {recall}%")
    print(f"  F1 Score:  {f1}%")

    # Feature importance
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_clf.fit(X_train, y_train)
    importances = dict(zip(
        feature_names,
        [round(float(x), 4) for x in rf_clf.feature_importances_]
    ))
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print("\n  Feature Importances:")
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"    {feat:20s} {bar} {imp:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    joblib.dump(xgb_clf, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classifier_scores': classifier_scores,
        'feature_importances': importances,
        'feature_names': feature_names,
        'confusion_matrix': cm.tolist(),
        'churn_rate': round(df['Exited'].mean() * 100, 1)
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  [OK] Saved: model.pkl, scaler.pkl, metrics.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
