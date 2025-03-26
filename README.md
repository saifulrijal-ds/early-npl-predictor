# Early NPL Predictor

A machine learning system for early detection of Non-Performing Loans (90+ days past due) in lending institutions.

## Overview

This project builds and deploys a machine learning model that predicts which loans will become 90+ days past due (NPL) within months 4-9 after origination, using only information available in the first 3 months. It implements a complete MLOps pipeline with continuous integration, continuous delivery, and automated deployment using BentoML.

## Key Features

- **Early Risk Detection**: Predicts NPLs 3-6 months before they occur
- **Automated MLOps Pipeline**: Uses GitHub Actions with CML for CI/CD
- **BentoML Deployment**: Packages and serves models as production-ready API services
- **Kubernetes Integration**: Enables scalable deployment to production environments
- **Realistic Synthetic Data**: Includes data generator for testing and development

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Kubernetes cluster (for production deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/early-npl-predictor.git
cd early-npl-predictor

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt