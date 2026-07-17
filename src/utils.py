"""Utility helpers for ChromaDB and other components of the recommendation system."""
import os
import pandas as pd
import chromadb
from google import genai
from google.genai import types
from chromadb import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv

load_dotenv()

def get_chroma_client(persist_directory):
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    return client

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


def get_or_create_collection(client, collection_name="samsung_phones"):
    """Gets or creates a ChromaDB collection using Gemini text embeddings."""
    emb_fn = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def create_phone_document(row):
    """Creates a rich natural language description of a phone based on its specs and scores for semantic indexing."""
    desc = (
        f"Samsung {row['Name']} ({row['Series']}) launched in {int(row['Launch_Year']) if not pd.isna(row['Launch_Year']) and row['Launch_Year'] != -1.0 else 'older years'}. "
        f"It is designed for the {row['Target_Segment']} target segment. "
        f"Hardware specs: {row['RAM_GB']:.0f}GB RAM, {row['Storage_GB']:.0f}GB storage, and a {row['Battery_mAh']:.0f}mAh battery capacity. "
        f"The screen display size is {row['Screen_Size_Inch']:.1f} inches with a {row['Refresh_Rate_Hz']:.0f}Hz refresh rate. "
        f"Camera hardware: {row['Main_Camera_MP']:.0f}MP Main Camera"
    )
    
    if not pd.isna(row['UltraWide_MP']) and row['UltraWide_MP'] > 0:
        desc += f", {row['UltraWide_MP']:.0f}MP Ultra-Wide camera"
    if not pd.isna(row['Telephoto_MP']) and row['Telephoto_MP'] > 0:
        desc += f", {row['Telephoto_MP']:.0f}MP Telephoto camera"
    if not pd.isna(row['Front_Camera_MP']) and row['Front_Camera_MP'] > 0:
        desc += f", and a {row['Front_Camera_MP']:.0f}MP front camera"
        
    desc += f". Features OIS support: {row['OIS']}, Waterproof: {row['Waterproof']}, Galaxy AI features: {row['AI_Features']}."
    
    desc += (
        f" Calculated Subsystem Scores (0-10): Performance: {row['performance_score']:.1f}, "
        f"Camera: {row['camera_score']:.1f}, Battery: {row['battery_score']:.1f}, "
        f"Display: {row['display_score']:.1f}, AI Score: {row['ai_score']:.1f}, "
        f"Durability: {row['durability_score']:.1f}. "
        f"Overall Machine Learning Recommendation Score: {row['recommendation_score']:.1f}."
    )
    return desc

def populate_chromadb(df, collection):
    """Populates ChromaDB with phone records from the DataFrame."""
    print("Formatting documents and metadata for ChromaDB...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        doc = create_phone_document(row)
        
        # Prepare metadata (ensure no NaN or float values that ChromaDB doesn't allow)
        # ChromaDB accepts strings, ints, floats, and bools in metadata
        metadata = {
            "name": str(row["Name"]),
            "series": str(row["Series"]),
            "target_segment": str(row["Target_Segment"]),
            "ram_gb": float(row["RAM_GB"]),
            "storage_gb": float(row["Storage_GB"]),
            "battery_mah": float(row["Battery_mAh"]),
            "screen_size_inch": float(row["Screen_Size_Inch"]),
            "refresh_rate_hz": float(row["Refresh_Rate_Hz"]),
            "main_camera_mp": float(row["Main_Camera_MP"]),
            "ultrawide_mp": float(row["UltraWide_MP"]) if not pd.isna(row["UltraWide_MP"]) else 0.0,
            "telephoto_mp": float(row["Telephoto_MP"]) if not pd.isna(row["Telephoto_MP"]) else 0.0,
            "front_camera_mp": float(row["Front_Camera_MP"]) if not pd.isna(row["Front_Camera_MP"]) else 0.0,
            "ois": str(row["OIS"]),
            "ai_features": str(row["AI_Features"]),
            "waterproof": str(row["Waterproof"]),
            "performance_score": float(row["performance_score"]),
            "camera_score": float(row["camera_score"]),
            "battery_score": float(row["battery_score"]),
            "display_score": float(row["display_score"]),
            "ai_score": float(row["ai_score"]),
            "durability_score": float(row["durability_score"]),
            "recommendation_score": float(row["recommendation_score"]),
            "launch_year": float(row["Launch_Year"]) if not pd.isna(row["Launch_Year"]) else -1.0,
            "launch_price": float(row["Launch_Price"]) if not pd.isna(row["Launch_Price"]) else -1.0
        }
        
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"phone_{idx}")

    print(f"Adding {len(documents)} phone records to ChromaDB collection...")
    # Add in batches to avoid size limit issues
    batch_size = 200
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
    print("ChromaDB population completed successfully!")

if __name__ == "__main__":
    print("utils.py is ready.")
