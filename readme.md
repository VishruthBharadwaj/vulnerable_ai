# ML Security Demo with Vertex AI Integration

This project demonstrates security vulnerabilities in ML pipelines based on the OWASP Top 10 and shows how to properly build and register models in Google Cloud's Vertex AI.

## Project Overview

This demonstration showcases:

1. A vulnerable ML pipeline with common security issues
2. A secure implementation with proper security practices
3. Integration with Google Cloud Vertex AI for model registry

## Security Vulnerabilities Demonstrated

The project simulates these OWASP Top 10 vulnerabilities:

1. **A1:2021 - Broken Access Control** : Demonstrating lack of access validation before model deployment
2. **A2:2021 - Cryptographic Failures** : Showing inadequate protection of sensitive data
3. **A3:2021 - Injection** : Simulating SQL injection in data processing
4. **A5:2021 - Security Misconfiguration** : Hard-coded credentials and paths
5. **A8:2021 - Software and Data Integrity Failures** : Insecure deserialization using pickle
6. **A9:2021 - Security Logging and Monitoring Failures** : Improper logging of sensitive information

## Secure Practices Demonstrated

The secure implementation shows:

1. Proper access control and role-based permissions
2. Secure data handling and serialization
3. Environment variable usage instead of hardcoded credentials
4. Proper logging practices
5. Secure model storage and deployment
6. Vertex AI integration for model registration

## Prerequisites

* Python 3.7+
* Google Cloud account with Vertex AI API enabled
* Required Python packages:
  * google-cloud-aiplatform
  * google-cloud-storage
  * scikit-learn
  * pandas
  * numpy
  * joblib

## Setup

1. Clone this repository:
   ```
   https://github.com/VishruthBharadwaj/vulnerable_ai
   cd vulnerable_ai
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up Google Cloud credentials:
   ```
   gcloud auth application-default login
   ```
4. Enable the Vertex AI API:
   ```
   gcloud services enable aiplatform.googleapis.com
   ```

## Granting Access to Collaborators in Vertex AI

To enable access for collaborators in Vertex AI:

1. Go to the Google Cloud Console
2. Navigate to IAM & Admin > IAM
3. Click "Add" to add a new principal
4. Enter the collaborator's email address
5. Assign one of these roles:
   * `Vertex AI User` - For using models and endpoints
   * `Vertex AI Admin` - For full access to all Vertex AI resources
   * `Vertex AI Developer` - For creating and managing models

## Running the Demo

Run the demo script with your project ID:

```bash
python demo_script.py --project-id YOUR_PROJECT_ID
```

Optional arguments:

* `--location`: Vertex AI location (default: us-central1)
* `--mode`: Which demo to run (choices: vulnerable, secure, both)
* `--role`: User role for testing access control

## Project Structure

* `vulnerable_ml_model.py`: Demonstrates ML pipeline with security vulnerabilities
* `secure_ml_vertex_integration.py`: Shows secure ML pipeline with Vertex AI integration
* `demo_script.py`: Script to run both demonstrations
* `requirements.txt`: Required Python packages

## Viewing Your Model in Vertex AI

After running the secure demo:

1. Go to Google Cloud Console
2. Navigate to Vertex AI > Models
3. Find your model in the list (named "text_classifier_[random_id]")

## Disclaimer

This project is for educational purposes only. The vulnerabilities are intentionally introduced for demonstration. Do not use the vulnerable code in a production environment.

## License

MIT License
