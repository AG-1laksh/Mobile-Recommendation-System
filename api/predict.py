"""Recommendation and prediction logic - api/ copy for Vercel serverless deployment."""
import os
import re
import pandas as pd
import numpy as np
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini client once at module level (warm requests reuse this)
_api_key = os.environ.get("GEMINI_API_KEY", "")
_genai_client = genai.Client(api_key=_api_key) if _api_key else None

from utils import get_chroma_client, get_or_create_collection

# Personas definition mapping
PERSONAS = {
    1: {
        "name": "The Power User (Tech-Enthusiast)",
        "goal": "Own the absolute fastest, most future-proof smartphone available.",
        "query": "fastest future-proof flagship smartphone with high performance processor, maximum RAM and storage, newest launch year",
        "filter": {"target_segment": "Flagship"}
    },
    2: {
        "name": "The Conscious Optimizer",
        "goal": "Find a highly capable device at the lowest possible price point.",
        "query": "affordable value for money budget smartphone with good features and high durability",
        "filter": {"target_segment": "Budget"}
    },
    3: {
        "name": "The Content Creator & Photographer",
        "goal": "Capture professional-grade photos and videos without carrying a heavy camera setup.",
        "query": "smartphone with best professional grade camera, high megapixels, telephoto optical zoom lens, ultra-wide and optical image stabilization OIS",
        "filter": {"target_segment": "Photography"}
    },
    4: {
        "name": "The Minimalist",
        "goal": "Purchase a reliable, uncomplicated device quickly without comparing complex technical specifications.",
        "query": "reliable simple uncomplicated all-rounder smartphone with great battery life and clear screen",
        "filter": {"target_segment": "Budget"}
    },
    5: {
        "name": "The Mobile Gamer",
        "goal": "Play graphics-heavy mobile games smoothly with zero lag or overheating issues.",
        "query": "gaming smartphone with high refresh rate screen, fastest performance GPU, large battery, zero lag cooling",
        "filter": {"target_segment": "Flagship"}
    },
    6: {
        "name": "The Feature-Driven Searcher",
        "goal": "Find a highly specific device that ticks exact, non-negotiable feature boxes.",
        "behavior": "Explicitly defines their own needs using filters, keywords, or text prompts rather than relying on past behavior.",
        "query": "",  # To be filled by user input
        "filter": None
    }
}

def extract_budget_from_query(query_str):
    text = query_str.lower().replace(",", "")
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*lakh', text)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', text)
    if k_match:
        return int(float(k_match.group(1)) * 1000)
    budget_keywords = ["under", "budget", "price", "below", "around", "less than", "within", "limit", "cost", "inr", "rs"]
    for kw in budget_keywords:
        pattern = rf"{kw}\s*(?:inr|rs\.?)?\s*(\d+)"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    nums = re.findall(r'\b\d{4,6}\b', text)
    if nums:
        return max(int(n) for n in nums)
    return None

def extract_year_preference(query_str):
    if not query_str:
        return None, False
    query_lower = query_str.lower()
    found_years = [int(yr) for yr in re.findall(r'\b\d{4}\b', query_lower)]
    if any(yr < 2024 for yr in found_years):
        return min(yr for yr in found_years if yr < 2024), True
    if any(kw in query_lower for kw in ["older", "old", "before 2024", "previous", "retro", "classic", "refurbished", "used"]):
        return None, True
    return None, False

def query_and_rank_recommendations(persona_id, user_query="", persist_dir=None):
    """Performs hybrid recommendation using ChromaDB and Gemini."""
    # On Vercel, __file__ points to api/predict.py, so data is at ../data/chroma_db
    if persist_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(script_dir, "../data/chroma_db")

    if persona_id not in PERSONAS:
        raise ValueError(f"Invalid persona ID: {persona_id}. Must be between 1 and 6.")

    persona = PERSONAS[persona_id]
    print(f"\n[Step 1] Selected Persona: {persona['name']}")

    search_query = persona["query"]
    metadata_filter = persona["filter"]

    if persona_id == 6:
        if not user_query:
            raise ValueError("For the Feature-Driven Searcher persona, a user query must be provided.")
        search_query = user_query

    _, asked_for_older = extract_year_preference(user_query if persona_id == 6 else "")

    print("\n[Step 2] Querying ChromaDB for candidate devices...")
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client)

    where_filter = None
    if not asked_for_older:
        year_filter = {"launch_year": {"$gte": 2024.0}}
        if metadata_filter:
            where_filter = {"$and": [metadata_filter, year_filter]}
        else:
            where_filter = year_filter
    else:
        where_filter = metadata_filter

    query_kwargs = {"query_texts": [search_query], "n_results": 100}
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    if not results or not results["metadatas"] or len(results["metadatas"][0]) == 0:
        print("No matches found. Retrying without filters...")
        results = collection.query(query_texts=[search_query], n_results=100)

    if not results or not results["metadatas"] or len(results["metadatas"][0]) == 0:
        return "Sorry, no Samsung phones were found in the database matching your criteria.", pd.DataFrame()

    candidates_metadata = results["metadatas"][0]
    candidates_docs = results["documents"][0]
    df_candidates = pd.DataFrame(candidates_metadata)
    df_candidates["document"] = candidates_docs

    if not asked_for_older and 'launch_year' in df_candidates.columns:
        df_candidates = df_candidates[df_candidates['launch_year'] >= 2024]

    budget_limit = None
    if persona_id == 6:
        budget_limit = extract_budget_from_query(search_query)
        if budget_limit is not None:
            print(f"Extracted budget limit from query: {budget_limit} INR")
            df_candidates = df_candidates[df_candidates['launch_price'] > 0]
            df_candidates = df_candidates[df_candidates['launch_price'] <= budget_limit]

    if df_candidates.empty:
        budget_str = f" of {budget_limit} INR" if budget_limit else ""
        return f"Sorry, no Samsung phones were found matching your criteria within your budget{budget_str}.", pd.DataFrame()

    print("\n[Step 3] Ranking candidates using recommendation score...")
    df_candidates_sorted = df_candidates.sort_values(by="recommendation_score", ascending=False)
    top_recommendations = df_candidates_sorted.head(3)

    context_list = []
    for i, (_, row) in enumerate(top_recommendations.iterrows()):
        launch_price_str = f"{int(row['launch_price'])} INR" if 'launch_price' in row and float(row['launch_price']) > 0 else "N/A"
        context_list.append(
            f"Device {i+1}: {row['name']}\n"
            f"- Specs: {row['ram_gb']:.0f}GB RAM, {row['storage_gb']:.0f}GB Storage, "
            f"{row['battery_mah']:.0f}mAh Battery, {row['screen_size_inch']:.1f}\" screen, "
            f"{row['refresh_rate_hz']:.0f}Hz refresh rate, {row['main_camera_mp']:.0f}MP Main Camera. "
            f"Launch Year: {int(row['launch_year']) if 'launch_year' in row and float(row['launch_year']) > 0 else 'Older'}.\n"
            f"- Price (INR Only): Launch Price: {launch_price_str}.\n"
            f"- Subsystem Scores: Performance: {row['performance_score']:.1f}/10, Camera: {row['camera_score']:.1f}/10, "
            f"Battery: {row['battery_score']:.1f}/10, Display: {row['display_score']:.1f}/10, AI: {row['ai_score']:.1f}/10, "
            f"Durability: {row['durability_score']:.1f}/10.\n"
            f"- Overall Recommendation Score: {row['recommendation_score']:.2f}/10.\n"
        )

    devices_context = "\n".join(context_list)

    llm_prompt = f"""
    You are an expert Samsung Mobile Recommendation Assistant.
    A user has selected the following persona:
    Persona: {persona['name']}
    Goal: {persona['goal']}

    {"User's Custom Query / Budget: " + search_query if persona_id == 6 else ""}
    {f"Note: The user specifies a budget of {budget_limit} INR. The recommended devices have been filtered to fit this budget based on their launch price." if budget_limit else ""}

    Based on their goal, here are the top 3 recommended Samsung devices retrieved from our system, ranked by our recommendation scoring engine:

    {devices_context}

    Please write a friendly, highly conversational, and expert recommendation explanation for this user.
    Address them directly according to their persona.
    For each device, explain clearly why it fits their specific goals and budget. Highlight prices (Launch Price in INR) and show how the specs directly map to their daily experience.
    Avoid comparing specs in a dry, technical table; explain the real-world value of these specs to their daily experience.
    Keep the tone premium, helpful, and concise. Make sure to structure the recommendations clearly.

    IMPORTANT: You MUST include this disclaimer at the end of your recommendation: "These phones are recommended on the basis of launch price"
    """

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        client = _genai_client or genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=llm_prompt,
            config=types.GenerateContentConfig(
                system_instruction='You are a helpful and polite Samsung phone recommender.'
            )
        )
        explanation = response.text
        disclaimer = "These phones are recommended on the basis of launch price"
        if disclaimer not in explanation:
            explanation += f"\n\nDisclaimer: {disclaimer}"
        return explanation, df_candidates_sorted
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        print("Falling back to a rules-based explanation template.")
        explanation = f"### Recommendations for {persona['name']}:\n\n"
        for i, (_, row) in enumerate(top_recommendations.iterrows()):
            launch_price_val = row.get('launch_price', -1.0)
            launch_price_str = f"{int(launch_price_val)} INR" if launch_price_val > 0 else "N/A"
            explanation += f"**{i+1}. {row['name']}** (Launch Price: {launch_price_str}, Score: {row['recommendation_score']:.2f}/10)\n"
            explanation += f"- Why it fits: This device offers strong {row['target_segment'].lower()} capabilities, a performance score of {row['performance_score']:.1f}/10, and a camera score of {row['camera_score']:.1f}/10.\n\n"
        explanation += "\nDisclaimer: These phones are recommended on the basis of launch price\n"
        return explanation, df_candidates_sorted
