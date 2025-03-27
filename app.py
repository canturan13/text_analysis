import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer, util
from starlette.responses import RedirectResponse
import torch
import fasttext


# Initialize logger with specified log level
logger = logging.getLogger("uvicorn.{}".format(os.getenv('LOG_LEVEL', 'info')))


# OpenAPI tags metadata
tags_metadata = [
    {
        "name": "Embedding",
        "description": "Generate Embeddings.",
    },
    {
        "name": "Language Detection",
        "description": "Detect the language of the input text.",
    },
    {
        "name": "Utilities",
        "description": "Utilities for the API.",
    },
]

# Create a FastAPI application
app = FastAPI(
    title="Text Analysis API",  # API title displayed in OpenAPI documentation
    version="0.0.1",  # API version from the environment variables
    openapi_tags=tags_metadata,  # OpenAPI tags for categorizing endpoints
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1
    },  # Swagger UI parameter to control models expansion depth
)

# Mount static files (if any) to the "/static" endpoint
app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
    ),
    name="static",
)


# Define a startup event handler for loading artifacts when the application starts
@app.on_event("startup")
def load_artifacts():
    """
    Load models for selected languages during application startup.
    """
    model_path = "models/jinaai-jina-embeddings-3"
    logger.info("Model path: %s", model_path)
    #check model path is valid
    if not os.path.exists(model_path):
        logger.error("Model path does not exist")
        model_path = "jinaai/jina-embeddings-v3"
    app.state.embedding = SentenceTransformer(model_path,trust_remote_code=True)
    app.state.embedding.eval()
    if torch.cuda.is_available():
      app.state.embedding.to(torch.device('cuda'))
      logger.info("Model loaded to CUDA")
    elif torch.backends.mps.is_available():
      app.state.embedding.to(torch.device('mps'))
      logger.info("Model loaded to MPS")
    else:
      app.state.embedding.to(torch.device('cpu'))
      logger.info("Model loaded to CPU")


    model_path = "models/fasttext"
    logger.info("Model path: %s", model_path)
    app.state.fasttext = fasttext.load_model(os.path.join(model_path, "lid.176.ftz"))

    logger.info("Model loaded successfully")

@app.post("/v1/embeddings", tags=["Embedding"], summary="Generate Embeddings")
def get_embeddings(sentences: List[str]):
    try:
        # Encode input sentences to get embeddings
        embeddings = app.state.embedding.encode(sentences)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}"
        )


@app.post("/v1/cosine-similarity", tags=["Embedding"], summary="Cosine Similarity")
def get_cosine_similarity(sentence1: str, sentence2: str):
    try:
        # Encode input sentences to get embeddings
        embedding1 = app.state.embedding.encode([sentence1])[0]
        embedding2 = app.state.embedding.encode([sentence2])[0]

        # Calculate cosine similarity
        similarity_score = util.cos_sim(embedding1, embedding2)
        return {"cosine_similarity": similarity_score.item()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating cosine similarity: {str(e)}"
        )

@app.post(
    "/detect",
    tags=["Language Detection"],
    summary="Detect the language of the input text",
)
async def detect_language(text: str):
    """
    Endpoint to detect the language of the input text using the 'fasttext' model.
    Args:
        request (Request): Raw request payload containing the input text.
    Returns:
        dict: Detection response containing language information.
    Raises:
        HTTPException: If an error occurs during language detection, return a 400 or 500 response.
    """
    try:
        if not text:
            raise HTTPException(
                status_code=400, detail="Input text cannot be empty."
            )

        labels, scores = app.state.fasttext.predict(text)
        label = labels[0].replace("__label__", "")
        score = min(float(scores[0]), 1.0)

        return {
            "text": text,
            "lang": label,
            "score": score,
        }
    except ValueError as ve:
        logger.error(f"ValueError during language detection: {ve}")
        raise HTTPException(
            status_code=400, detail=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error occurred during language detection: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error occurred during language detection: {str(e)}"
        )

@app.get(
    "/health",
    tags=["Utilities"],
    summary="Check if the API is running",
)
def health():
    """
    Health check endpoint to verify if the API is running.
    Returns:
        dict: Status message indicating the API is okay.
    """
    return {"status": "ok"}

@app.get(
    "/",
    include_in_schema=False,
)
def main():
    """
    Redirects the root URL to the API documentation (FastAPI Swagger UI).
    Returns:
        RedirectResponse: Redirects to the API documentation.
    """
    return RedirectResponse(url="/docs")


# Define a custom OpenAPI schema function
def custom_openapi():
    """
    Custom function to generate OpenAPI schema for the Embedding API.
    Returns:
        dict: OpenAPI schema for the API.
    """
    openapi_schema = get_openapi(
        title="Text Analysis API",
        version="0.0.1",
        description="Embedding Generation",
        routes=app.routes,
        tags=tags_metadata,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the default OpenAPI schema with the custom function
app.openapi = custom_openapi

# Add CORS (Cross-Origin Resource Sharing) middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn app:app --host 0.0.0.0 --port 8000