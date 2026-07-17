import os
import sys

# ─── CRITICAL: These patches must happen before ANY other import ───────────────

# 1. Patch sqlite3 with pysqlite3-binary (chromadb requires SQLite >= 3.35)
#    Vercel's Lambda environment has an older version, so we override it.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Running locally without pysqlite3 is fine

# 2. Set CHROMA_HOME to /tmp so the ONNX embedding model can be
#    downloaded and cached in Vercel's writable /tmp directory.
#    Without this, chromadb tries to write to ~/.cache which is read-only.
os.environ.setdefault('CHROMA_HOME', '/tmp/chroma_cache')

# ─── Now safe to import everything else ───────────────────────────────────────

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Add the api/ directory to sys.path so sibling modules (predict.py, utils.py)
# can be imported by Python without any path hacks.
api_dir = os.path.dirname(os.path.abspath(__file__))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from predict import query_and_rank_recommendations, PERSONAS

# Static files (HTML/CSS/JS) live in FRONTEND/ at the repo root
frontend_dir = os.path.abspath(os.path.join(api_dir, '..', 'FRONTEND'))

app = Flask(__name__, static_folder=frontend_dir, static_url_path='')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json or {}
        persona_id = data.get('persona_id')
        user_query = data.get('query', '')

        if persona_id is None:
            return jsonify({"error": "persona_id is required"}), 400
        try:
            persona_id = int(persona_id)
        except ValueError:
            return jsonify({"error": "persona_id must be an integer"}), 400
        if persona_id not in PERSONAS:
            return jsonify({"error": "Invalid persona ID. Must be 1-6"}), 400

        # Point to chroma_db relative to the repo root
        persist_dir = os.path.abspath(os.path.join(api_dir, '..', 'data', 'chroma_db'))

        explanation, df_candidates = query_and_rank_recommendations(persona_id, user_query, persist_dir=persist_dir)
        explanation = explanation.replace(
            "These phones are recommended on the basis of launch price",
            "Recommendations may vary. Verify specifications before purchase."
        )

        candidates_list = []
        if df_candidates is not None and not df_candidates.empty:
            df_cleaned = df_candidates.where(pd.notnull(df_candidates), None)
            candidates_list = df_cleaned.to_dict(orient='records')

        top_recommendations = candidates_list[:3]

        persona_info = {
            "name": PERSONAS[persona_id]["name"],
            "goal": PERSONAS[persona_id]["goal"],
            "id": persona_id
        }

        return jsonify({
            "explanation": explanation,
            "recommendations": top_recommendations,
            "candidates": candidates_list,
            "persona": persona_info
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Samsung Mobile Recommendation Server on http://localhost:{port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
