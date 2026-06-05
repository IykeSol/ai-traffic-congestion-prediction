"""
AI Traffic Congestion Predictor
Flask REST API Backend + Gemini NLP Integration

Endpoints:
  POST /api/predict         — structured feature prediction
  POST /api/nlp-predict     — natural language → Gemini → ML → summary
  GET  /api/model-stats     — accuracy, F1, feature importances
  GET  /api/history         — last 20 predictions (session memory)
  GET  /                    — serves the web UI
"""

import os, sys, json, math, time, traceback
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

from ml.model import TrafficPredictor

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
METRICS_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "ml", "models", "metrics.json")

# In-memory prediction history (last 50)
prediction_history: deque = deque(maxlen=50)

# ─────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────────────────────
try:
    predictor = TrafficPredictor.load("traffic_predictor")
    print("[INFO] Model loaded successfully.")
except Exception as e:
    predictor = None
    print(f"[WARNING] Model not loaded: {e}. Run python ml/train.py first.")

# Configure Gemini
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use gemini-flash-latest which is available for this API key
        gemini_model = genai.GenerativeModel("gemini-flash-latest")
        print("[INFO] Gemini API configured.")
    except Exception as e:
        print(f"[WARNING] Gemini setup failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sincos(val, maxval):
    angle = 2 * math.pi * val / maxval
    return round(math.sin(angle), 5), round(math.cos(angle), 5)


def _build_full_features(raw: dict) -> dict:
    """Compute cyclical features and fill defaults from raw input dict."""
    hour        = int(raw.get("hour", 8))
    dow         = int(raw.get("day_of_week", 0))
    month       = int(raw.get("month", 6))
    is_weekend  = int(dow >= 5)

    hs, hc = _sincos(hour, 24)
    ds, dc = _sincos(dow, 7)
    ms, mc = _sincos(month, 12)

    return {
        "hour":                    hour,
        "hour_sin":                hs,
        "hour_cos":                hc,
        "day_of_week":             dow,
        "day_sin":                 ds,
        "day_cos":                 dc,
        "month":                   month,
        "month_sin":               ms,
        "month_cos":               mc,
        "is_weekend":              is_weekend,
        "road_type":               str(raw.get("road_type", "arterial")),
        "city_zone":               str(raw.get("city_zone", "CBD")),
        "weather_condition":       str(raw.get("weather_condition", "clear")),
        "temperature_c":           float(raw.get("temperature_c", 25.0)),
        "vehicle_count":           int(raw.get("vehicle_count", 1500)),
        "average_speed_kmh":       float(raw.get("average_speed_kmh", 50.0)),
        "signal_cycle_time":       float(raw.get("signal_cycle_time", 90.0)),
        "road_capacity_utilization": float(raw.get("road_capacity_utilization", 0.60)),
        "incident_reported":       int(raw.get("incident_reported", 0)),
        "school_zone":             int(raw.get("school_zone", 0)),
        "construction_zone":       int(raw.get("construction_zone", 0)),
        "public_event_nearby":     int(raw.get("public_event_nearby", 0)),
    }


def _store_history(entry: dict):
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_history.appendleft(entry)


EXTRACT_PROMPT = """
You are an AI assistant that extracts structured traffic features from a natural language description.

From the user description below, extract these fields and return ONLY valid JSON:
- hour (int, 0-23)
- day_of_week (int, 0=Monday to 6=Sunday)
- month (int, 1-12)
- road_type (one of: highway, arterial, local, expressway)
- city_zone (one of: CBD, residential, industrial, suburban, airport)
- weather_condition (one of: clear, rain, fog, snow, storm)
- temperature_c (float)
- vehicle_count (int, estimated vehicles per hour)
- average_speed_kmh (float)
- signal_cycle_time (float, seconds, default 90)
- road_capacity_utilization (float, 0.0 to 1.0)
- incident_reported (int, 0 or 1)
- school_zone (int, 0 or 1)
- construction_zone (int, 0 or 1)
- public_event_nearby (int, 0 or 1)

Use reasonable defaults for anything not mentioned. Current time context: {time_context}.

User description: "{description}"

Return ONLY the JSON object, no explanation, no markdown fences.
"""

SUMMARY_PROMPT = """
You are an expert urban traffic analyst AI. A machine learning model has predicted the following traffic condition:

- Predicted congestion level: {prediction}
- Confidence: {confidence}%
- Road conditions provided: {conditions_summary}

Write a concise, professional 3-sentence analysis that:
1. Explains what this congestion level means for commuters
2. Identifies the likely primary contributing factors based on the inputs
3. Gives 2 specific, actionable recommendations

Write in a formal but accessible tone. No bullet points, just flowing prose. No emojis.
"""


def call_gemini_extract(description: str) -> dict:
    """Use Gemini to extract structured features from free text."""
    if not gemini_model:
        raise RuntimeError("Gemini API not configured. Add GEMINI_API_KEY to .env")

    now = datetime.now()
    time_ctx = now.strftime("%A, %B %Y, %H:%M")
    prompt = EXTRACT_PROMPT.format(
        description=description,
        time_context=time_ctx,
    )
    response = gemini_model.generate_content(prompt)
    text = response.text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    return json.loads(text)


def call_gemini_summarize(prediction: str, confidence: float,
                          conditions: dict) -> str:
    """Use Gemini to generate a natural language summary of the prediction."""
    if not gemini_model:
        return (
            f"The model predicts {prediction.upper()} congestion with "
            f"{confidence:.1f}% confidence based on current road conditions."
        )

    cond_parts = []
    for k, v in conditions.items():
        if k not in ("hour_sin","hour_cos","day_sin","day_cos",
                     "month_sin","month_cos"):
            cond_parts.append(f"{k.replace('_',' ')}: {v}")
    conditions_summary = ", ".join(cond_parts[:10])

    prompt = SUMMARY_PROMPT.format(
        prediction=prediction,
        confidence=round(confidence, 1),
        conditions_summary=conditions_summary,
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded. Run python ml/train.py first."}), 503

    try:
        raw = request.get_json(force=True)
        features = _build_full_features(raw)
        result   = predictor.predict_single(features)

        ai_summary = ""
        try:
            ai_summary = call_gemini_summarize(result["prediction"], result["top_confidence"], raw)
        except Exception as e:
            ai_summary = "AI analysis could not be generated at this time."
            print(f"[WARNING] Gemini summary failed: {e}")

        entry = {
            "input":       features,
            "prediction":  result["prediction"],
            "confidence":  result["top_confidence"],
            "source":      "structured",
        }
        _store_history(entry)

        return jsonify({
            "success":    True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top_confidence": result["top_confidence"],
            "ai_summary": ai_summary
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/nlp-predict", methods=["POST"])
def nlp_predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        data        = request.get_json(force=True)
        description = data.get("description", "").strip()
        if not description:
            return jsonify({"error": "No description provided."}), 400

        # Step 1: Gemini extracts features
        raw_features = call_gemini_extract(description)

        # Step 2: Build full feature dict
        features = _build_full_features(raw_features)

        # Step 3: ML prediction
        result = predictor.predict_single(features)

        # Step 4: Gemini summarizes
        summary = call_gemini_summarize(
            result["prediction"],
            result["top_confidence"],
            features,
        )

        entry = {
            "input":       features,
            "prediction":  result["prediction"],
            "confidence":  result["top_confidence"],
            "source":      "nlp",
            "description": description,
            "summary":     summary,
        }
        _store_history(entry)

        return jsonify({
            "success":        True,
            "description":    description,
            "extracted":      raw_features,
            "prediction":     result["prediction"],
            "confidence":     result["confidence"],
            "top_confidence": result["top_confidence"],
            "summary":        summary,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Gemini returned invalid JSON: {e}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-stats", methods=["GET"])
def model_stats():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        else:
            metrics = {
                "accuracy": 0, "f1_macro": 0, "f1_weighted": 0,
                "feature_importances": {},
                "classes": ["low", "moderate", "high", "severe"],
            }

        gemini_active = gemini_model is not None
        model_loaded  = predictor is not None

        return jsonify({
            "success":       True,
            "metrics":       metrics,
            "gemini_active": gemini_active,
            "model_loaded":  model_loaded,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 20))
    return jsonify({
        "success": True,
        "history": list(prediction_history)[:limit],
        "total":   len(prediction_history),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": predictor is not None,
        "gemini":       gemini_model is not None,
        "timestamp":    datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  AI Traffic Congestion Predictor API")
    print(f"  Running on http://0.0.0.0:{port}")
    print(f"  Gemini: {'Active' if gemini_model else 'Not configured (add GEMINI_API_KEY to .env)'}")
    print(f"  Model:  {'Loaded' if predictor else 'Not found — run python ml/train.py'}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
