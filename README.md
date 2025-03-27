# Text Analysis

This project provides tools and models for text analysis, including language detection and text embedding.

## Features

- **Language Detection**: Detects the language of a given text using the FastText model.
- **Text Embedding**: Generates embeddings for text using Sentence Transformers.

## Requirements

- Python 3.8 or higher
- `pip` package manager

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/canturan13/text_analysis.git
   cd text_analysis
	 pip install -r requirements.txt
	 uvicorn app:app
	 # or set host and port
	 uvicorn app:app --host 0.0.0.0 --port 8000
	 ```
