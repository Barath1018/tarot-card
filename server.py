import os
import pickle
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file from the same directory as server.py
load_dotenv(Path(__file__).parent / ".env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")

app = Flask(__name__)
CORS(app)

# --- Gemini AI Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print(f"[Oracle] Gemini AI connected successfully.")
else:
    gemini_model = None
    print(f"[Oracle] No API key found. Add GEMINI_API_KEY to your .env file.")


def normalize_gesture_label(label):
    normalized = str(label).strip().lower().replace("_", "-").replace(" ", "-")

    aliases = {
        "palm": "open-palm",
        "openhand": "open-palm",
        "open-hand": "open-palm",
        "openpalm": "open-palm",
        "open-palm": "open-palm",
        "backhand": "back-hand",
        "back-of-hand": "back-hand",
        "back-hand": "back-hand",
        "pointright": "point-right",
        "point-right": "point-right",
        "pointleft": "point-left",
        "point-left": "point-left",
        "thumbs": "thumbs-up",
        "thumbsup": "thumbs-up",
        "thumbs-up": "thumbs-up",
        "fist": "fist",
    }

    return aliases.get(normalized, normalized)


with open(MODEL_PATH, "rb") as model_file:
    gesture_model = pickle.load(model_file)


@app.route("/")
@app.route("/Mystic_Tarot.html")
def index():
    return send_from_directory(BASE_DIR, "Mystic_Tarot.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks")

    if not isinstance(landmarks, list) or len(landmarks) != 42:
        return jsonify({"error": "landmarks must be a list of 42 floats"}), 400

    try:
        features = [[float(value) for value in landmarks]]
    except (TypeError, ValueError):
        return jsonify({"error": "landmarks must contain numeric values"}), 400

    prediction = gesture_model.predict(features)[0]
    return jsonify({"gesture": normalize_gesture_label(prediction)})


@app.route("/oracle", methods=["POST"])
def oracle():
    """Gemini AI tarot/celestial reading endpoint."""
    if not gemini_model:
        return jsonify(
            {
                "error": "GEMINI_API_KEY not configured. Set it as an environment variable."
            }
        ), 503

    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "tarot")
    cards = payload.get("cards", [])

    if not cards:
        return jsonify({"error": "No cards provided"}), 400

    try:
        card = cards[0]
        if mode == "celestial":
            prompt = (
                f"You are a cosmic oracle channeling celestial energies. The seeker has connected with {card['name']}. "
                f"Its energy speaks: {card['meaning']}. "
                f"Give a short, mystical and poetic reading in 3-4 sentences that captures cosmic wisdom. "
                f"Speak directly to the seeker using 'you'. Do not use bullet points or headers. "
                f"Make it feel vast, infinite, and profound like the cosmos itself."
            )
        else:
            prompt = (
                f"You are a mystical tarot oracle. The seeker has drawn the {card['name']} card. "
                f"Its core meaning is: {card['meaning']}. "
                f"Give a short, deeply personal and poetic reading in 3-4 sentences. "
                f"Speak directly to the seeker using 'you'. Do not use bullet points or headers. "
                f"Make it feel magical and profound."
            )

        response = gemini_model.generate_content(prompt)
        reading_text = response.text.strip()
        return jsonify({"reading": reading_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
