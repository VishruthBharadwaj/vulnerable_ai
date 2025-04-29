import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import tempfile
import json
from google.cloud import aiplatform

# VULNERABILITY 1: Insecure deserialization - using pickle without validation
# For demonstration purposes only - in production, use safer alternatives

# Define malicious payload at module level to allow pickling
def malicious_code(x):
    print("Malicious code executed!")
    return x

def load_data_insecure(filename):
    """
    VULNERABILITY: Insecure Deserialization (A8:2021)
    Loads data using pickle without validation
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)  # Insecure - could execute arbitrary code if pickle is malicious
    else:
        # For demo, create synthetic data if file doesn't exist
        texts = [
            "This is a positive review of the product",
            "I really like this service",
            "The experience was terrible",
            "Very disappointed with the quality",
            "Amazing product, would recommend",
        ]
        labels = [1, 1, 0, 0, 1]  # 1 for positive, 0 for negative
        return texts, labels

def query_data_vulnerable(user_input):
    """
    VULNERABILITY: Injection (A3:2021)
    Simulates SQL injection by directly using user input in a query
    """
    query = f"SELECT * FROM user_data WHERE user_input = '{user_input}'"
    print(f"Executing query: {query}")  # Shows how injection could occur
    return [f"Result for {user_input}"]

def process_user_data(user_data, api_key):
    """
    VULNERABILITY: Logging and Monitoring Failures (A9:2021)
    Logs sensitive information
    """
    print(f"Processing data with API key: {api_key}")  # Sensitive data exposure
    return user_data

def build_text_classifier():
    """
    Creates a simple text classification model
    """
    texts, labels = load_data_insecure("training_data.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    return pipeline

def save_model(model, path="model.pkl", cloud_storage="gs://vulnerable-bucket"):
    """
    VULNERABILITY: Security Misconfiguration (A5:2021) and Software and Data Integrity Failures (A8:2021)
    Hard-coded credentials and paths, plus simulated insecure deserialization with a detectable payload.
    """
    cloud_credentials = {
        "api_key": "AIzaSyC_fake_key_for_demonstration_only",
        "secret": "s3cr3t_k3y_d3m0_only"
    }
    # Attach the malicious payload (now globally defined)
    model._malicious_attr = malicious_code
    # Save locally with pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved locally to {path}")
    print(f"Would upload to {cloud_storage} using {cloud_credentials['api_key']}")
    return path

def deploy_model_vulnerable(model_path, user_role="user"):
    """
    VULNERABILITY: Broken Access Control (A1:2021)
    No proper access validation before deployment
    """
    print(f"User with role '{user_role}' attempting to deploy model")
    print("Deployment proceeding without proper authorization checks")
    print(f"Model from {model_path} deployed successfully")

def main():
    print("Building model...")
    model = build_text_classifier()
    api_key = "sk_live_51HFakeKeyForDemo123456789"
    process_user_data({"user": "demo"}, api_key)
    model_path = save_model(model)
    deploy_model_vulnerable(model_path, user_role="guest")
    print("\nVulnerability demonstration complete.")
    print("In a real-world scenario, this code would contain multiple security issues.")

if __name__ == "__main__":
    main()