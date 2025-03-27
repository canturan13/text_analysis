# Text Analysis

This project provides tools and models for text analysis, including language detection and text embedding.

## Features

- **Text Embedding**: Generates embeddings for text using Sentence Transformers.

## Requirements

- Python 3.8 or higher
- `pip` package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/canturan13/text_analysis.git
   cd text_analysis
	 ```

### Using Poetry

1. Install dependencies using Poetry:
   ```bash
	poetry install
	 ```

2. Activate the virtual environment:
   ```bash
	poetry shell
	 ```

3. Run the application:
   ```bash
	poetry run uvicorn app:app --host 0.0.0.0 --port 8000
	 ```


### Using Pip

1. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

2. Run the application:
	 ```bash
	 uvicorn app:app --host 0.0.0.0 --port 8000
	 ```

### Using Python Env

1. Create a virtual environment:
	 ```bash
	 python -m venv venv
	 source venv/bin/activate
	 ```
2. Install dependencies:

	 ```bash
	 pip install --upgrade setuptools wheel
	 pip install -r requirements.txt
	 ```

3. Run the application:
	 ```bash
	 uvicorn app:app --host 0.0.0.0 --port 8000
	 ```