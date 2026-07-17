"""Utility helpers for ChromaDB - Vercel serverless copy (uses Gemini embeddings, no ONNX model)."""
import os
from google import genai
from google.genai import types
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function using Google Gemini text-embedding-004 model.
    Replaces the heavy local ONNX model (all-MiniLM-L6-v2) to eliminate cold-start
    model downloads in serverless environments like Vercel.
    """
    def __init__(self, api_key: str = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY must be set to use GeminiEmbeddingFunction.")
        self._client = genai.Client(api_key=self._api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of text documents using the Gemini API."""
        embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


def get_chroma_client(persist_directory):
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def get_or_create_collection(client, collection_name="samsung_phones"):
    """Gets or creates a ChromaDB collection using Gemini text embeddings."""
    emb_fn = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection
