# Vertex AI Demo Project

A Python project for working with Google Cloud Vertex AI, including model training, deployment, and prediction capabilities.

## Features

- Model training and evaluation
- Model deployment to Vertex AI
- Batch and online predictions
- Custom training with Python packages
- Integration with Google Cloud services

## Prerequisites

- Python 3.8 or higher
- Google Cloud SDK installed and configured
- Google Cloud project with Vertex AI API enabled
- Service account with appropriate permissions

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Set up your Google Cloud credentials:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   ```
2. Set your project ID:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

## Usage

### Training a Model

```bash
python src/training/train_model.py
```

### Making Predictions

```bash
python src/prediction/predict.py
```

## Project Structure

```
vertexai_demo/
├── src/
│   ├── training/          # Model training scripts
│   ├── prediction/        # Prediction utilities
│   ├── data/             # Data processing utilities
│   └── utils/            # Common utilities
├── config/               # Configuration files
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for exploration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## License

MIT License
