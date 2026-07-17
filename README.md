# Minor Project Lab

## Project Structure
```text
Minor_proj_lab/
├── data/                  # Raw, processed, and external datasets
├── notebooks/             # Jupyter notebooks for exploration and prototyping
├── src/                   # Source code package
│   ├── graph/             # Graph construction and manipulation utilities
│   ├── features/          # Feature engineering and preprocessing
│   ├── models/            # Model architectures and training loops
│   └── eval/              # Evaluation metrics and validation scripts
├── tests/                 # Unit and integration tests
├── experiments/           # Experiment configuration, logs, and outputs
├── requirements.txt       # Pinned Python dependencies
└── README.md              # Project documentation
```

## Setup & Installation
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
