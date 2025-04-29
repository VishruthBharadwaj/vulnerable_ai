#!/usr/bin/env python3

import os
import argparse
import time
from google.cloud import aiplatform

# Import our modules
import sys
sys.path.append('.')
from vulnerable_ml_model import main as run_vulnerable_demo
from secure_ml_vertex_integration import main as run_secure_demo

# Set environment variable for authentication
def setup_authentication(key_path='service-account-key.json'):
    """Set up authentication using service account key"""
    # Check if key file exists
    if not os.path.exists(key_path):
        print(f"Error: Service account key file {key_path} not found.")
        print("Please make sure the key file is in the correct location.")
        return False
    
    # Set environment variable to point to the key file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
    print(f"Authentication configured using service account key: {key_path}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='ML Security Demo')
    parser.add_argument('--project-id', type=str, default='sec-eng-414005',
                      help='Your Google Cloud Project ID (default: sec-eng-414005)')
    parser.add_argument('--location', type=str, default='us-central1',
                      help='Vertex AI location (default: us-central1)')
    parser.add_argument('--mode', type=str, choices=['vulnerable', 'secure', 'both'],
                      default='both', help='Which demo to run')
    parser.add_argument('--role', type=str, default='admin',
                      help='User role for testing access control')
    return parser.parse_args()

def run_vulnerable_demonstration():
    print("\n" + "="*80)
    print(" VULNERABLE ML DEMONSTRATION - DO NOT USE THESE PRACTICES IN PRODUCTION ")
    print("="*80 + "\n")
    
    print("Running vulnerable ML pipeline to demonstrate OWASP Top 10 vulnerabilities...")
    run_vulnerable_demo()
    
    print("\n" + "="*80)
    print(" END OF VULNERABLE DEMONSTRATION ")
    print("="*80 + "\n")

def run_secure_demonstration(project_id, location, role):
    print("\n" + "="*80)
    print(" SECURE ML IMPLEMENTATION WITH VERTEX AI INTEGRATION ")
    print("="*80 + "\n")
    
    print(f"Running secure ML pipeline with project: {project_id}, location: {location}, role: {role}...")
    
    # Set environment variables for the demo
    os.environ["MODEL_STORAGE_BUCKET"] = f"{project_id}-ml-models"
    
    # Run secure implementation
    model = run_secure_demo(project_id, role)
    
    if model:
        print(f"\nModel successfully registered in Vertex AI!")
        print(f"Model ID: {model.resource_name}")
        print(f"Model display name: {model.display_name}")
        print(f"Model URI: {model.artifact_uri}")
        
        # Print instructions for viewing the model
        print("\nYou can view your model in the Vertex AI Model Registry:")
        print(f"https://console.cloud.google.com/vertex-ai/models?project={project_id}")
    else:
        print("\nModel registration failed. Please check the logs for details.")
    
    print("\n" + "="*80)
    print(" END OF SECURE DEMONSTRATION ")
    print("="*80 + "\n")

def main():
    args = parse_args()
    
    print("ML Security Demo - Comparing Vulnerable vs Secure Implementations")
    print("This demo shows OWASP Top 10 vulnerabilities in ML pipelines and how to fix them")
    
    # Set up authentication
    if not setup_authentication():
        print("Authentication setup failed. Exiting.")
        return
        
    # Run selected demos
    if args.mode in ['vulnerable', 'both']:
        run_vulnerable_demonstration()
        
    if args.mode in ['secure', 'both']:
        run_secure_demonstration(args.project_id, args.location, args.role)
    
    print("\nDemo complete!")
    print("For educational purposes only. Always follow security best practices in production.")

if __name__ == "__main__":
    main()