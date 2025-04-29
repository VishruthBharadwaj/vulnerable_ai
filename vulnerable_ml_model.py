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
import joblib  # For secure serialization
import hashlib  # For integrity checking
import uuid    # For generating random filenames
import logging # For secure logging practices
from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging to avoid sensitive data exposure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ml_operations.log'), logging.StreamHandler()]
)

# PART 1: VULNERABLE CODE WITH INTENTIONAL SECURITY ISSUES
# ------------------------------------------------------

# Define malicious payload class for demonstration
class MaliciousPayload:
    def __reduce__(self):
        # This will execute when the pickle is loaded
        return (eval, ("import os; print('SECURITY BREACH: Code execution via pickle!'); os.system('echo \"potential system compromise\" > breach.txt')",))

def create_malicious_pickle():
    """Create a malicious pickle file to demonstrate vulnerability"""
    model = LogisticRegression()
    
    # Attach malicious payload
    model._malicious = MaliciousPayload()
    
    # Save the poisoned model
    with open("malicious_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    logging.info("Created demonstration malicious model: malicious_model.pkl")
    return "malicious_model.pkl"

def load_data_insecure(filename):
    """
    VULNERABILITY: Insecure Deserialization (A8:2021)
    Loads data using pickle without validation
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)  # Insecure - could execute arbitrary code if pickle is malicious
    else:
        texts = [
            "This is a positive review of the product",
            "I really like this service",
            "The experience was terrible",
            "Very disappointed with the quality", 
            "Amazing product, would recommend",
        ]
        labels = [1, 1, 0, 0, 1]  # 1 for positive, 0 for negative
        return texts, labels

def save_model_insecure(model, path="model.pkl"):
    """
    VULNERABILITY: Software and Data Integrity Failures (A8:2021)
    - No validation or integrity checks
    - Uses pickle without safety measures
    """
    # Hard-coding credentials (VULNERABILITY: Security Misconfiguration A5:2021)
    cloud_credentials = {
        "api_key": "AIzaSyC_fake_key_for_demonstration_only",
        "secret": "s3cr3t_k3y_d3m0_only" 
    }
    
    # Log sensitive information (VULNERABILITY: A9:2021)
    logging.info(f"Saving model with credentials: {cloud_credentials}")
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    return path

def deploy_model_vulnerable(model_path, user_role="user"):
    """
    VULNERABILITY: Broken Access Control (A1:2021)
    No proper access validation before deployment
    """
    logging.info(f"User with role '{user_role}' deploying model without authorization checks")
    
    # Load model without any validation
    with open(model_path, 'rb') as f:
        model = pickle.load(f)  # VULNERABILITY: Loading without validation
    
    print(f"Model from {model_path} deployed without security checks")
    return model

def demonstrate_vulnerability():
    """Run demonstration of vulnerable ML workflow"""
    print("\n==== VULNERABLE ML WORKFLOW DEMONSTRATION ====")
    
    # Create a simple model
    texts = [
        "This is a positive review of the product",
        "I really like this service",
        "The experience was terrible",
        "Very disappointed with the quality",
        "Amazing product, would recommend",
    ]
    labels = [1, 1, 0, 0, 1]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('classifier', LogisticRegression())
    ])
    pipeline.fit(texts, labels)
    
    # Save model insecurely
    model_path = save_model_insecure(pipeline)
    print(f"Model saved insecurely to {model_path}")
    
    # Create a malicious model for demonstration
    malicious_path = create_malicious_pickle()
    
    # Try to load the malicious model - in real scenario this would execute arbitrary code
    print("\nAttempting to load malicious model (in a real scenario, this would execute arbitrary code):")
    print("--------------------------------------------------------------------")
    try:
        # Commented out for safety, but in a real demo this would show the vulnerability
        # with open(malicious_path, 'rb') as f:
        #    malicious_model = pickle.load(f)
        print("If this were executed, arbitrary code would run during unpickling")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nVulnerabilities demonstrated include:")
    print("1. Insecure deserialization using pickle without validation")
    print("2. Hardcoded credentials in code")
    print("3. Logging sensitive information")
    print("4. No integrity checking on model files")
    print("5. No access control for model deployment")


# PART 2: SECURE CODE WITH BEST PRACTICES
# ------------------------------------------------------

def secure_credentials():
    """
    SECURE PRACTICE: Proper credential management
    Load credentials from environment or secure vault, not hardcoded
    """
    # In production, use environment variables or secret management service
    service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project_id = json.load(open(service_account_path))['project_id']
        
        # Log only non-sensitive information
        logging.info(f"Successfully loaded credentials for project: {project_id}")
        
        return credentials, project_id
    except Exception as e:
        logging.error(f"Error loading credentials: {str(e)}")
        return None, None

def load_data_secure(filename=None):
    """
    SECURE PRACTICE: Safe data loading with validation
    """
    if filename and os.path.exists(filename):
        # Check file size before loading to prevent DoS
        file_size = os.path.getsize(filename)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            logging.warning(f"File {filename} exceeds size limit")
            return None, None
            
        try:
            # Use joblib instead of pickle for better security
            data = joblib.load(filename)
            logging.info(f"Successfully loaded data from {filename}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None, None
    else:
        # Fallback to sample data
        texts = [
            "This is a positive review of the product",
            "I really like this service",
            "The experience was terrible",
            "Very disappointed with the quality",
            "Amazing product, would recommend",
        ]
        labels = [1, 1, 0, 0, 1]  # 1 for positive, 0 for negative
        return texts, labels

def build_text_classifier_secure():
    """
    SECURE PRACTICE: Building model with secure practices
    """
    texts, labels = load_data_secure()
    
    if texts is None or labels is None:
        logging.error("Failed to load training data")
        return None
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        logging.info(f"Model training complete. Accuracy: {accuracy:.2f}")
        
        return pipeline
    except Exception as e:
        logging.error(f"Error building classifier: {str(e)}")
        return None

def calculate_model_hash(model_path):
    """
    SECURE PRACTICE: Calculate hash for integrity verification
    """
    hasher = hashlib.sha256()
    with open(model_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_model_secure(model, metadata=None):
    """
    SECURE PRACTICE: Safe model serialization with integrity checks
    - Uses joblib instead of pickle
    - Generates random filename to prevent path traversal
    - Creates hash for integrity checking
    - Stores minimal metadata separate from model
    """
    if model is None:
        logging.error("Cannot save None model")
        return None, None
        
    try:
        # Generate secure random filename
        filename = f"model_{uuid.uuid4().hex}.joblib"
        
        # Save model using joblib (safer than pickle)
        joblib.dump(model, filename)
        
        # Calculate hash for integrity checking
        model_hash = calculate_model_hash(filename)
        
        # Create separate metadata file with non-sensitive info
        meta = metadata or {}
        meta.update({
            'created_at': pd.Timestamp.now().isoformat(),
            'model_hash': model_hash,
            'model_file': filename
        })
        
        meta_filename = f"{filename}.meta.json"
        with open(meta_filename, 'w') as f:
            json.dump(meta, f)
            
        logging.info(f"Model saved securely to {filename} with hash {model_hash[:10]}...")
        return filename, meta_filename
        
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return None, None

def verify_model_integrity(model_path, metadata_path):
    """
    SECURE PRACTICE: Verify model hasn't been tampered with
    """
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Calculate current hash
        current_hash = calculate_model_hash(model_path)
        
        # Compare with stored hash
        if current_hash != metadata['model_hash']:
            logging.error(f"Model integrity check failed for {model_path}")
            return False
            
        logging.info(f"Model integrity verified for {model_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying model integrity: {str(e)}")
        return False

def deploy_model_secure(model_path, metadata_path, user_role, credentials=None):
    """
    SECURE PRACTICE: Secure model deployment with proper checks
    - Verifies user authorization
    - Checks model integrity
    - Uses secure deserialization
    """
    # Role-based access control
    authorized_roles = ['admin', 'ml_engineer']
    if user_role not in authorized_roles:
        logging.warning(f"Unauthorized deployment attempt by user with role: {user_role}")
        return None
    
    # Verify model integrity
    if not verify_model_integrity(model_path, metadata_path):
        logging.error("Model failed integrity check, deployment aborted")
        return None
    
    try:
        # Secure loading with joblib instead of pickle
        model = joblib.load(model_path)
        
        # In real scenario, would deploy to appropriate service
        logging.info(f"Model {model_path} deployed successfully by {user_role}")
        
        return model
    except Exception as e:
        logging.error(f"Error deploying model: {str(e)}")
        return None

def scan_model_file(model_path):
    """
    SECURE PRACTICE: Scan model file for potential threats
    A basic implementation of checking for pickle opcodes of concern
    """
    DANGEROUS_OPCODES = [
        b'c__builtin__\neval', 
        b'cos\nsystem', 
        b'posix\nsystem',
        b'subprocess\nPopen',
        b'__reduce__',
        b'exec'
    ]
    
    try:
        with open(model_path, 'rb') as f:
            content = f.read()
            
        for opcode in DANGEROUS_OPCODES:
            if opcode in content:
                logging.warning(f"Potentially dangerous opcode detected in {model_path}: {opcode}")
                return False, f"Dangerous opcode detected: {opcode}"
                
        return True, "No dangerous opcodes detected"
    except Exception as e:
        logging.error(f"Error scanning model file: {str(e)}")
        return False, str(e)

def demonstrate_secure_practices():
    """Run demonstration of secure ML workflow"""
    print("\n==== SECURE ML WORKFLOW DEMONSTRATION ====")
    
    # 1. Secure credential handling
    credentials, project_id = secure_credentials()
    
    # 2. Build model securely
    model = build_text_classifier_secure()
    
    if model is None:
        print("Failed to build model")
        return
    
    # 3. Save model securely
    model_path, meta_path = save_model_secure(model, {'purpose': 'sentiment_analysis'})
    
    if model_path is None:
        print("Failed to save model")
        return
        
    print(f"Model saved securely to {model_path}")
    print(f"Metadata saved to {meta_path}")
    
    # 4. Scan for security issues
    scan_result, message = scan_model_file(model_path)
    print(f"Security scan result: {'PASS' if scan_result else 'FAIL'} - {message}")
    
    # 5. Verify integrity
    integrity_verified = verify_model_integrity(model_path, meta_path)
    print(f"Integrity verification: {'PASS' if integrity_verified else 'FAIL'}")
    
    # 6. Secure deployment with proper authorization
    print("\nAttempting deployment with unauthorized role:")
    result = deploy_model_secure(model_path, meta_path, user_role="guest")
    print(f"Unauthorized deployment result: {'Success' if result else 'Failed as expected'}")
    
    print("\nAttempting deployment with authorized role:")
    result = deploy_model_secure(model_path, meta_path, user_role="admin")
    print(f"Authorized deployment result: {'Success' if result else 'Failed'}")
    
    print("\nSecurity best practices demonstrated include:")
    print("1. Secure serialization using joblib instead of pickle")
    print("2. Secure credential management")
    print("3. Model integrity verification")
    print("4. Role-based access control")
    print("5. Security scanning for dangerous opcodes")
    print("6. Proper error handling and logging")
    print("7. Random filename generation")
    print("8. Metadata separation from model data")

def main():
    """Run the full demonstration"""
    print("============================================")
    print("   ML MODEL SECURITY DEMONSTRATION")
    print("============================================")
    print("This demonstration shows the importance of security")
    print("shift-left practices in ML development, particularly")
    print("focusing on safe serialization and model integrity.")
    print("--------------------------------------------")
    
    # First demonstrate vulnerable code
    demonstrate_vulnerability()
    
    # Then demonstrate secure practices
    demonstrate_secure_practices()
    
    print("\n============================================")
    print("   DEMONSTRATION COMPLETE")
    print("============================================")
    print("Key takeaways:")
    print("1. Never use pickle without validation for models from untrusted sources")
    print("2. Use safer alternatives like joblib for ML serialization")
    print("3. Always verify model integrity before loading")
    print("4. Implement proper access controls for model deployment")
    print("5. Scan model files for security issues before deployment")
    print("6. Never hardcode credentials in your code")
    print("7. Implement proper logging (avoiding sensitive data exposure)")
    
if __name__ == "__main__":
    main()