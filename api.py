"""
BridgeAI — Flask Backend API
Rulează cu: python api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import score_products
from main import extract_intent

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:8081",
    "http://localhost:19006",
    "http://localhost:3000",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:19006",
])

# Translation map for user-facing category names  
CATEGORY_EN = {
    "laptop":           "laptop",
    "smartphone":       "smartphone",
    "monitor":          "monitor",
    "tv":               "TV",
    "tableta":          "tablet",
    "casti":            "headphones",
    "casti gaming":     "gaming headphones",
}

def _cat_en(categorie: str) -> str:
    return CATEGORY_EN.get(categorie, categorie)
 
def _parse_intent(user_message: str) -> dict:
    """Call extract_intent — returns Romanian keys directly, no remapping needed."""
    raw = extract_intent(user_message, [])
    return {
        "categorie":   raw.get("categorie", "laptop"),
        "scop":        raw.get("scop", "casual"),
        "buget":       float(raw.get("buget", 0)),
        "brand":       raw.get("brand", ""),
        "clarificare": raw.get("clarificare", ""),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "BridgeAI API"}), 200


@app.route("/nlp", methods=["POST"])
def nlp():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing field 'message'"}), 400
    try:
        intent = _parse_intent(data["message"])
        return jsonify(intent), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    required = ["categorie", "scop", "buget"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field '{field}'"}), 400

    intent = {
        "categorie": data.get("categorie", "laptop"),
        "scop":      data.get("scop", "casual"),
        "buget":     float(data.get("buget", 0)),
        "brand":     data.get("brand", ""),
    }

    try:
        products = score_products(intent, top_n=int(data.get("top_n", 5)))
        return jsonify(products), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing field 'message'"}), 400

    # Step 1: NLP
    try:
        intent = _parse_intent(data["message"])
    except Exception as e:
        return jsonify({"error": f"NLP error: {e}"}), 500

    # Step 2: Ask for budget if missing
    if not intent.get("buget") or intent["buget"] == 0:
        clarification = f"What budget do you have in mind for the {_cat_en(intent.get('categorie', 'device'))} (in lei)?"
        return jsonify({
            "intent": intent,
            "products": [],
            "needs_budget": True,
            "clarification": clarification,
        }), 200

    # Step 3: RF scoring
    try:
        products = score_products(intent, top_n=5)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"RF error: {e}"}), 500

    return jsonify({
        "intent": intent,
        "products": products,
        "needs_budget": False,
        "clarification": "",
    }), 200


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 BridgeAI API — starting server")
    print("="*55)
    print("  GET  http://localhost:5000/health")
    print("  POST http://localhost:5000/nlp")
    print("  POST http://localhost:5000/recommend")
    print("  POST http://localhost:5000/chat")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)