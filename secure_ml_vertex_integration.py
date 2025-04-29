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
import json
from google.oauth2 import service_account

# Configure proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('secure_ml_pipeline')

# SECURE PRACTICE 1: Safer serialization with checksums
def save_model_secure(model, base_path="model"):
    """
    Securely save model using joblib instead of pickle
    Also creates a checksum file for integrity verification
    """
    # Create unique filename
    unique_id = uuid.uuid4().hex[:8]
    model_path = f"{base_path}_{unique_id}.joblib"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Calculate and save checksum
    checksum = calculate_file_hash(model_path)
    checksum_path = f"{model_path}.sha256"
    
    with open(checksum_path, 'w') as f:
        f.write(checksum)
    
    # Create metadata file
    metadata = {
        "model_id": unique_id,
        "model_type": "text_classifier",
        "framework": "scikit-learn",
        "checksum": checksum,
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = f"{model_path}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved securely to {model_path} with checksum {checksum[:10]}...")
    
    return model_path, checksum_path, metadata_path

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of file for integrity verification"""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

# SECURE PRACTICE 2: Parameterized queries
def query_data_secure(user_input: str) -> List[str]:
    """
    Demonstrates secure parameterized queries
    In a real system, you would use database parameterization
    """
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
    
    # Log without exposing sensitive data - use hash of user ID
    user_id_hash = hashlib.sha256(str(user_data.get('user', 'unknown')).encode()).hexdigest()[:10]
    logger.info(f"Processing data for user ID hash: {user_id_hash}")
    
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

# Initialize GCS client
def initialize_gcs_client(service_account_path=None):
    """Initialize GCS client with proper authentication"""
    try:
        if service_account_path:
            # Use specific service account file if provided
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path
            )
            client = storage.Client(credentials=credentials)
            
            # Log project from service account (not credentials themselves)
            project_id = json.load(open(service_account_path))['project_id']
            logger.info(f"Initialized GCS client with service account for project: {project_id}")
        else:
            # Use default credentials
            client = storage.Client()
            logger.info(f"Initialized GCS client with default credentials")
            
        return client
    except Exception as e:
        pass

# Function to upload model to GCS
def upload_model_to_gcs(local_model_path, local_checksum_path, local_metadata_path, 
                        gcs_path, client):
    """Upload model files to Google Cloud Storage"""
    # Extract bucket_name and blob_name from gcs_path
    gcs_path = gcs_path.replace("gs://", "")
    if "/" in gcs_path:
        bucket_name, base_folder = gcs_path.split("/", 1)
    else:
        bucket_name = gcs_path
        base_folder = ""
    
    # Create a unique subdirectory name - use the same ID from the model filename
    model_id = os.path.basename(local_model_path).split("_")[-1].split(".")[0]
    model_dir = f"model_{model_id}"
    
    if base_folder:
        gcs_model_dir = f"{base_folder}/{model_dir}"
    else:
        gcs_model_dir = model_dir
    
    logger.info(f"Uploading model to gs://{bucket_name}/{gcs_model_dir}")
    
    
    # Try to get bucket
    try:
        bucket = client.get_bucket(bucket_name)
        logger.info(f"Using existing bucket: {bucket_name}")
    except Exception as e:
        logger.info(f"Creating new bucket: {bucket_name}")
        bucket = client.create_bucket(bucket_name, location="us-central1")
    
    # Upload model file
    model_blob_name = f"{gcs_model_dir}/{os.path.basename(local_model_path)}"
    model_blob = bucket.blob(model_blob_name)
    model_blob.upload_from_filename(local_model_path)
    logger.info(f"Uploaded model file to gs://{bucket_name}/{model_blob_name}")
    
    # Upload checksum file
    checksum_blob_name = f"{gcs_model_dir}/{os.path.basename(local_checksum_path)}"
    checksum_blob = bucket.blob(checksum_blob_name)
    checksum_blob.upload_from_filename(local_checksum_path)
    logger.info(f"Uploaded checksum file to gs://{bucket_name}/{checksum_blob_name}")
    
    # Upload metadata file
    metadata_blob_name = f"{gcs_model_dir}/{os.path.basename(local_metadata_path)}"
    metadata_blob = bucket.blob(metadata_blob_name)
    metadata_blob.upload_from_filename(local_metadata_path)
    logger.info(f"Uploaded metadata file to gs://{bucket_name}/{metadata_blob_name}")
    
    # Return the GCS directory path which contains all model files
    return f"gs://{bucket_name}/{gcs_model_dir}"
    


# Function to register model in Vertex AI
def register_model_vertex_ai(
    gcs_model_dir_path: str, 
    project_id: str, 
    location: str = "us-central1",
    display_name: str = "text_classifier"
):
    """Register model to Vertex AI Model Registry"""
    
    # Initialize Vertex AI SDK with project ID
    aiplatform.init(project=project_id, location=location)
    
    # Extract model ID from path
    model_id = gcs_model_dir_path.split('/')[-1].split('_')[-1]
    
    # Create a unique display name with model ID
    unique_name = f"{display_name}_{model_id}"
    
    logger.info(f"Registering model in Vertex AI with name: {unique_name}")
    logger.info(f"Using artifact URI: {gcs_model_dir_path}")
    
    # First verify the model files exist in GCS
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_model_dir_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_model_dir_path.replace(f"gs://{bucket_name}/", "").split("/"))
    
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    if not blobs:
        logger.error(f"No files found at {gcs_model_dir_path}. Cannot register model.")
        return None
        
    logger.info(f"Found {len(blobs)} files in {gcs_model_dir_path}")
    for blob in blobs:
        logger.info(f"  - {blob.name}")
    
    # Upload the model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=unique_name,
        artifact_uri=gcs_model_dir_path,
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
    

# Main execution function
def main(project_id: str, service_account_path: str = None, user_role: str = "admin", location: str = "us-central1"):
    """Main pipeline function"""
    try:
        logger.info(f"Starting secure ML pipeline with project ID: {project_id}")

        # Create and train the model
        logger.info("Building secure model...")
        model = build_text_classifier()

        # Securely save the model locally with checksums and metadata
        model_path, checksum_path, metadata_path = save_model_secure(model)

        # Initialize GCS client
        gcs_client = initialize_gcs_client(service_account_path)

        # Define GCS folder path
        gcs_base_path = get_cloud_storage_path(project_id)

        # Check permissions and authorize deployment
        deployment_result = deploy_model_secure(gcs_base_path, user_role, project_id)
        if not deployment_result["success"]:
            logger.error(f"Deployment authorization failed: {deployment_result['message']}")
            return None
            
        # Upload model, checksum and metadata to GCS
        try:
            gcs_model_dir = upload_model_to_gcs(
                model_path, 
                checksum_path, 
                metadata_path, 
                gcs_base_path, 
                gcs_client
            )
            logger.info(f"All model files uploaded to {gcs_model_dir}")
        except Exception as e:
            logger.error(f"Failed to upload model files to GCS: {str(e)}")
            return None

        # Register in Vertex AI
        try:
            model = register_model_vertex_ai(gcs_model_dir, project_id, location)
            if model:
                logger.info("Model successfully built, deployed, and registered in Vertex AI")
                logger.info(f"Model resource name: {model.resource_name}")
                return model
            else:
                logger.error("Failed to register model in Vertex AI")
                return None
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            return None

    except Exception as e:
        pass

if __name__ == "__main__":
    # Use service account for authentication
    SERVICE_ACCOUNT_PATH = "service-account-key.json"
    PROJECT_ID = "sec-eng-414005"  # This should match the project_id in the service account JSON
    
    try:
        model = main(PROJECT_ID, SERVICE_ACCOUNT_PATH)
        if model:
            print(f"Pipeline completed successfully!")
        else:
            print(f"Pipeline failed. Check logs for details.")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")