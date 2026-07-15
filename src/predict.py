"""Recommendation and prediction logic combining XGBoost ranking and ChromaDB retrieval."""
import os
import sys
import re
import pandas as pd
import numpy as np
import joblib
import chromadb
import ollama

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
        "filter": {"target_segment": "Budget"} # Fallback search filter
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
    """
    Extracts a numeric budget in INR from a user query string.
    Supports formats like 'under 50000', 'budget of 30k', 'below 1.2 Lakhs', etc.
    """
    text = query_str.lower().replace(",", "")
    
    # 1. Check for lakh patterns
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*lakh', text)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)
        
    # 2. Check for K (thousands) patterns
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', text)
    if k_match:
        return int(float(k_match.group(1)) * 1000)
        
    # 3. Check for numeric patterns near budget keywords
    budget_keywords = ["under", "budget", "price", "below", "around", "less than", "within", "limit", "cost", "inr", "rs"]
    for kw in budget_keywords:
        pattern = rf"{kw}\s*(?:inr|rs\.?)?\s*(\d+)"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
            
    # 4. Fallback to any 4-6 digit number in the query
    nums = re.findall(r'\b\d{4,6}\b', text)
    if nums:
        return max(int(n) for n in nums)
        
    return None

def load_xgboost_model(model_path):
    """Loads the trained XGBoost model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}. Please run train_model.py first.")
    return joblib.load(model_path)

def query_and_rank_recommendations(persona_id, user_query="", persist_dir=None, model_path=None):
    """
    Performs hybrid recommendation:
    1. Retrieval from ChromaDB using semantic query & metadata filtering.
    2. Filtering by budget (comparing with Launch_Price) if user specified one.
    3. Scoring & Ranking retrieved candidates using XGBoost.
    4. Conversational explanation generation using Llama 3.2.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if persist_dir is None:
        persist_dir = os.path.join(script_dir, "../data/chroma_db")
    if model_path is None:
        model_path = os.path.join(script_dir, "../data/models/xgboost_model.joblib")

    # 1. Load Persona
    if persona_id not in PERSONAS:
        raise ValueError(f"Invalid persona ID: {persona_id}. Must be between 1 and 6.")
    
    persona = PERSONAS[persona_id]
    print(f"\n[Step 1] Selected Persona: {persona['name']}")
    print(f"Goal: {persona['goal']}")
    
    # Define actual query and filters
    search_query = persona["query"]
    metadata_filter = persona["filter"]
    
    if persona_id == 6:
        if not user_query:
            raise ValueError("For the Feature-Driven Searcher persona, a user query must be provided.")
        search_query = user_query
        print(f"User Query: {user_query}")
        
    # Check if user query asks for phones before 2024
    asked_for_older = False
    if persona_id == 6 and user_query:
        query_lower = user_query.lower()
        # Find any 4-digit years in the query
        found_years = [int(yr) for yr in re.findall(r'\b\d{4}\b', query_lower)]
        if any(yr < 2024 for yr in found_years):
            asked_for_older = True
        elif any(kw in query_lower for kw in ["older", "old", "before 2024", "previous", "retro", "classic", "refurbished", "used"]):
            asked_for_older = True

    # 2. Query ChromaDB
    print("\n[Step 2] Querying ChromaDB for candidate devices...")
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client)
    
    # Combine metadata filter and launch year filter
    where_filter = None
    if not asked_for_older:
        year_filter = {"launch_year": {"$gte": 2024.0}}
        if metadata_filter:
            where_filter = {
                "$and": [
                    metadata_filter,
                    year_filter
                ]
            }
        else:
            where_filter = year_filter
    else:
        where_filter = metadata_filter

    # Retrieve top 100 candidates to ensure we have enough after budget filtering
    query_kwargs = {
        "query_texts": [search_query],
        "n_results": 100
    }
    if where_filter:
        query_kwargs["where"] = where_filter
        
    results = collection.query(**query_kwargs)
    
    if not results or not results["metadatas"] or len(results["metadatas"][0]) == 0:
        print("No matches found in vector database. Retrying without filters...")
        results = collection.query(query_texts=[search_query], n_results=100)
        
    if not results or not results["metadatas"] or len(results["metadatas"][0]) == 0:
        return "Sorry, no Samsung phones were found in the database matching your criteria."
        
    candidates_metadata = results["metadatas"][0]
    candidates_docs = results["documents"][0]
    
    # Create candidates DataFrame
    df_candidates = pd.DataFrame(candidates_metadata)
    df_candidates["document"] = candidates_docs
    
    # Extra safety check: Filter by launch year in Python if not asked for older
    if not asked_for_older and 'launch_year' in df_candidates.columns:
        df_candidates = df_candidates[df_candidates['launch_year'] >= 2024]
        
    # Apply filtering for Persona 6 budget if specified
    budget_limit = None
    if persona_id == 6:
        budget_limit = extract_budget_from_query(search_query)
        if budget_limit is not None:
            print(f"Extracted budget limit from query: {budget_limit} INR")
            # Filter out N/A or invalid launch price (which have launch_price <= 0)
            df_candidates = df_candidates[df_candidates['launch_price'] > 0]
            # Filter by budget (compare launch_price <= budget_limit)
            df_candidates = df_candidates[df_candidates['launch_price'] <= budget_limit]
            print(f"Filtered candidate count within budget: {len(df_candidates)}")
            
    if df_candidates.empty:
        budget_str = f" of {budget_limit} INR" if budget_limit else ""
        return f"Sorry, no Samsung phones were found in the database matching your criteria within your budget{budget_str}."
        
    # 3. Predict & Rank using XGBoost
    print("\n[Step 3] Ranking candidates using the trained XGBoost model...")
    model = load_xgboost_model(model_path)
    
    # Format candidates for prediction
    features = [
        "ram_gb", "storage_gb", "battery_mah", "screen_size_inch", "refresh_rate_hz",
        "main_camera_mp", "ultrawide_mp", "telephoto_mp", "front_camera_mp", "launch_year"
    ]
    
    # Re-map ChromaDB metadata keys to XGBoost feature names
    feature_mapping = {
        "ram_gb": "RAM_GB",
        "storage_gb": "Storage_GB",
        "battery_mah": "Battery_mAh",
        "screen_size_inch": "Screen_Size_Inch",
        "refresh_rate_hz": "Refresh_Rate_Hz",
        "main_camera_mp": "Main_Camera_MP",
        "ultrawide_mp": "UltraWide_MP",
        "telephoto_mp": "Telephoto_MP",
        "front_camera_mp": "Front_Camera_MP",
        "launch_year": "Launch_Year"
    }
    
    df_candidates_mapped = df_candidates.rename(columns=feature_mapping)
    
    # Handle placeholder launch years
    df_candidates_mapped["Launch_Year"] = df_candidates_mapped["Launch_Year"].replace(-1.0, np.nan)
    
    # Select only required features
    X_candidates = df_candidates_mapped[[feature_mapping[f] for f in features]]
    
    # Predict scores
    predicted_scores = model.predict(X_candidates)
    
    # Add predicted scores back and sort
    df_candidates["predicted_score"] = predicted_scores
    df_candidates_sorted = df_candidates.sort_values(by="predicted_score", ascending=False)
    
    # Keep top 3 recommendations
    top_recommendations = df_candidates_sorted.head(3)
    
    print("\nTop 3 Ranked Recommendations:")
    for i, (_, row) in enumerate(top_recommendations.iterrows()):
        launch_price_val = row.get('launch_price', -1.0)
        launch_price_str = f"{int(launch_price_val)} INR" if launch_price_val > 0 else "N/A"
        print(f"{i+1}. {row['name']} (Launch Price: {launch_price_str}, XGBoost Predicted Score: {row['predicted_score']:.2f})")
    print("Disclaimer: This phones recommended on the basis of launch price")
        
    # 4. Generate response using Llama 3.2
    print("\n[Step 4] Requesting Llama 3.2 to generate explanation...")
    
    # Construct context for Llama 3.2
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
            f"- Overall Machine Learning Recommendation Score: {row['predicted_score']:.2f}/10.\n"
        )
        
    devices_context = "\n".join(context_list)
    
    llm_prompt = f"""
    You are an expert Samsung Mobile Recommendation Assistant.
    A user has selected the following persona:
    Persona: {persona['name']}
    Goal: {persona['goal']}
    
    {"User's Custom Query / Budget: " + search_query if persona_id == 6 else ""}
    {f"Note: The user specifies a budget of {budget_limit} INR. The recommended devices have been filtered to fit this budget based on their launch price." if budget_limit else ""}
    
    Based on their goal, here are the top 3 recommended Samsung devices retrieved from our system, ranked by our machine learning scoring engine:
    
    {devices_context}
    
    Please write a friendly, highly conversational, and expert recommendation explanation for this user.
    Address them directly according to their persona.
    For each device, explain clearly why it fits their specific goals and budget. Highlight prices (Launch Price in INR) and show how the specs directly map to their daily experience. Do NOT mention or use Current Price.
    Avoid comparing specs in a dry, technical table; explain the real-world value of these specs to their daily experience.
    Keep the tone premium, helpful, and concise. Make sure to structure the recommendations clearly.
    
    IMPORTANT: You MUST include this disclaimer at the end of your recommendation: "This phones recommended on the basis of launch price"
    """
    
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'system', 'content': 'You are a helpful and polite Samsung phone recommender.'},
            {'role': 'user', 'content': llm_prompt}
        ])
        explanation = response['message']['content']
        disclaimer = "This phones recommended on the basis of launch price"
        if disclaimer not in explanation:
            explanation += f"\n\nDisclaimer: {disclaimer}"
        return explanation
    except Exception as e:
        print(f"Error calling local Llama 3.2 via Ollama: {e}")
        print("Falling back to a rules-based explanation template.")
        # Fallback text if LLM call fails
        explanation = f"### Recommendations for {persona['name']}:\n\n"
        for i, (_, row) in enumerate(top_recommendations.iterrows()):
            launch_price_val = row.get('launch_price', -1.0)
            launch_price_str = f"{int(launch_price_val)} INR" if launch_price_val > 0 else "N/A"
            explanation += f"**{i+1}. {row['name']}** (Launch Price: {launch_price_str}, Score: {row['predicted_score']:.2f}/10)\n"
            explanation += f"- Why it fits: This device offers strong {row['target_segment'].lower()} capabilities, a performance score of {row['performance_score']:.1f}/10, and a camera score of {row['camera_score']:.1f}/10.\n\n"
        explanation += "\nDisclaimer: This phones recommended on the basis of launch price\n"
        return explanation

def interactive_cli():
    print("=" * 60)
    print("SAMSUNG SMARTPHONE RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("Choose a persona to receive recommendations:")
    for key, p in PERSONAS.items():
        print(f"{key}. {p['name']} - Goal: {p['goal']}")
        
    try:
        choice = int(input("\nEnter choice (1-6): ").strip())
        if choice not in PERSONAS:
            print("Invalid choice.")
            return
            
        user_query = ""
        if choice == 6:
            user_query = input("Describe what features or budget you are looking for (e.g. 'under 30000 INR', '12GB RAM'): ").strip()
            if not user_query:
                print("Query cannot be empty.")
                return
                
        explanation = query_and_rank_recommendations(choice, user_query)
        print("\n" + "=" * 60)
        print("RECOMMENDATION DECISION REPORT (Llama 3.2 + XGBoost)")
        print("=" * 60)
        print(explanation)
        print("=" * 60)
        
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) > 1:
        # Programmatic CLI parsing
        try:
            pid = int(sys.argv[1])
            q = sys.argv[2] if len(sys.argv) > 2 else ""
            resp = query_and_rank_recommendations(pid, q)
            print(resp)
        except Exception as ex:
            print(f"Error: {ex}")
    else:
        interactive_cli()
