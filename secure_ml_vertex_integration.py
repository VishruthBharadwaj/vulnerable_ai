import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib  
from google.cloud import aiplatform
from google.cloud import storage
import logging
import uuid
import re
from typing import List, Tuple, Dict, Any
import hashlib
import hmac
import secrets

# Configure proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('secure_ml_pipeline')

# SECURE PRACTICE 1: Safer serialization
def save_model_secure(model, path="model.joblib"):
    """Securely save model using joblib instead of pickle"""
    joblib.dump(model, path)
    logger.info(f"Model saved securely to {path}")
    return path

# SECURE PRACTICE 2: Parameterized queries
def query_data_secure(user_input: str) -> List[str]:
    """
    Demonstrates secure parameterized queries
    In a real system, you would use database parameterization
    """

    
    # For our demo, we'll just validate and return synthetic data
    # Sanitize input - only allow alphanumeric and spaces
    sanitized_input = re.sub(r'[^\w\s]', '', user_input)
    logger.info(f"Sanitized input: {sanitized_input}")
    
    # For demo, return synthetic data
    return [f"Result for {sanitized_input}"]

# SECURE PRACTICE 3: Secure logging
def process_user_data_secure(user_data: Dict, secret_manager=None):
    """Process user data without logging sensitive information"""

    # For demo, we'll simulate retrieving an API key securely
    if secret_manager:
        api_key = secret_manager.get_secret("api_key")
    else:
        api_key = "**simulated_retrieval_from_secret_manager**"
    
    # Log without exposing sensitive data
    logger.info(f"Processing data for user ID: {hash(str(user_data.get('user', 'unknown')))}")
    
    # Processing would happen here
    return user_data

# SECURE PRACTICE 4: Environment variables and secret management
def get_cloud_storage_path(project_id: str):
    """Get cloud storage path from environment variables"""
    # Get bucket name from environment or use project ID with suffix
    bucket_name = os.environ.get("MODEL_STORAGE_BUCKET", f"{project_id}-ml-models")
    folder = "text_classifier"
    return f"gs://{bucket_name}/{folder}"

# SECURE PRACTICE 5: Proper access control
def deploy_model_secure(model_path: str, user_role: str, project_id: str):
    """Implement proper access control before deployment"""
    # Define roles with permissions
    role_permissions = {
        "admin": ["read", "write", "deploy"],
        "developer": ["read", "write"],
        "user": ["read"]
    }
    
    # Check if user has deployment permission
    if "deploy" not in role_permissions.get(user_role, []):
        logger.warning(f"Access denied: User with role '{user_role}' attempted to deploy model")
        return {"success": False, "message": "Access denied: Insufficient permissions"}
    
    logger.info(f"Authorized deployment by user with role '{user_role}'")
    

    # For our demo, we'll just return success
    return {"success": True, "message": f"Model from {model_path} deployed successfully"}

# Function to create secure synthetic data
def create_synthetic_data():
    """Create synthetic training data"""
    texts = [
        "This is a positive review of the product",
        "I really like this service",
        "The experience was terrible",
        "Very disappointed with the quality",
        "Amazing product, would recommend",
        "This product exceeded my expectations",
        "Not worth the money, don't buy it",
        "Customer service was unhelpful",
        "Great value for the price",
        "Would purchase again"
    ]
    labels = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  # 1 for positive, 0 for negative
    return texts, labels

# Build a simple model
def build_text_classifier():
    """Creates a simple text classification model with proper practices"""
    # Create synthetic data
    texts, labels = create_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create a pipeline with TF-IDF and logistic regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.2f}")
    
    return pipeline

# Function to upload model to GCS
def upload_model_to_gcs(local_model_path: str, gcs_path: str, project_id: str):
    """Upload model file to Google Cloud Storage"""
    # Extract bucket_name and blob_name from gcs_path
    gcs_path = gcs_path.replace("gs://", "")
    bucket_name, blob_name = gcs_path.split("/", 1)

    # Create a unique subdirectory name
    model_dir_name = f"model_{uuid.uuid4().hex[:8]}" # Unique directory name
    gcs_model_dir_path = f"{blob_name}/{model_dir_name}"  # Path to the new directory

    # Initialize storage client with project from service account
    storage_client = storage.Client(project=project_id)

    try:
        # Try to get bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
            logger.info(f"Using existing bucket: {bucket_name}")
        except Exception as e:
            logger.info(f"Creating new bucket: {bucket_name}")
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")

        # Upload file to the new directory
        blob = bucket.blob(f"{gcs_model_dir_path}/{os.path.basename(local_model_path)}") #Upload to the directory
        blob.upload_from_filename(local_model_path)
        logger.info(f"Model uploaded to gs://{bucket_name}/{gcs_model_dir_path}/{os.path.basename(local_model_path)}")

        return f"gs://{bucket_name}/{gcs_model_dir_path}" 

    except Exception as e:
        logger.error(f"Error uploading model to GCS: {str(e)}")
        raise

# Function to register model in Vertex AI
def register_model_vertex_ai(
    gcs_model_path: str, 
    project_id: str, 
    location: str = "us-central1",
    display_name: str = "text_classifier"
):
    """Register model to Vertex AI Model Registry"""
    try:
        # Initialize Vertex AI SDK with project ID from service account
        aiplatform.init(project=project_id, location=location)
        
        # Create a unique display name with timestamp
        unique_name = f"{display_name}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Registering model in Vertex AI with name: {unique_name}")
        
        # Upload the model to Vertex AI Model Registry
        model = aiplatform.Model.upload(
            display_name=unique_name,
            artifact_uri=gcs_model_path,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest",
            description="Text classification model for sentiment analysis",
            labels={
                "framework": "scikit-learn",
                "task": "text-classification",
                "algorithm": "logistic-regression"
            }
        )
        
        logger.info(f"Model registered in Vertex AI. Model ID: {model.resource_name}")
        return model
    
    except Exception as e:
        logger.error(f"Error registering model in Vertex AI: {str(e)}")
        return None

# Main execution function

def main(project_id: str, user_role: str = "admin", location: str = "us-central1"):
    """Main pipeline function"""
    try:
        logger.info(f"Starting secure ML pipeline with project ID: {project_id}")

        # Create and train the model
        logger.info("Building secure model...")
        model = build_text_classifier()

        # Securely save the model locally
        model_path = save_model_secure(model)

        # Define GCS path
        gcs_folder = get_cloud_storage_path(project_id)

        # Upload model to GCS and get the directory path
        try:
            full_gcs_path = upload_model_to_gcs(model_path, gcs_folder, project_id)
            logger.info(f"Model uploaded to {full_gcs_path}")
        except Exception as e:
            logger.error(f"Failed to upload model to GCS: {str(e)}")
            return None

        # Check permissions and deploy
        deployment_result = deploy_model_secure(full_gcs_path, user_role, project_id)

        if deployment_result["success"]:
            # Register in Vertex AI
            model = register_model_vertex_ai(full_gcs_path, project_id, location)
            if model:
                logger.info("Model successfully built, deployed, and registered in Vertex AI")
                return model
            else:
                logger.error("Failed to register model in Vertex AI")
                return None
        else:
            logger.error(f"Deployment failed: {deployment_result['message']}")
            return None

    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}")
        return None


if __name__ == "__main__":
    # This would use the project ID from the service account when run via demo_script.py
    # But for direct execution, default to the project in the service account
    PROJECT_ID = "sec-eng-414005"
    model = main(PROJECT_ID)