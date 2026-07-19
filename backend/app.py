"""
CareLens AI — Flask backend
-----------------------------
Exposes the AI matching engine as an API. Frontend (vanilla JS) calls this.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from triage_engine import TriageEngine
from ocr_engine import MedicineScanner

app = Flask(__name__)
CORS(app)  # allow the frontend (different origin once deployed) to call this API

engine = TriageEngine()
scanner = MedicineScanner()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB


@app.route("/", methods=["GET"])
def health_check():
    """Simple route to confirm the API is alive — useful for testing the
    deployed link and for judges checking the backend responds at all."""
    return jsonify({"status": "ok", "service": "CareLens AI backend"})


@app.route("/assess-symptoms", methods=["POST"])
def assess_symptoms():
    data = request.get_json(silent=True) or {}
    symptom_text = data.get("symptoms", "").strip()

    if not symptom_text:
        return jsonify({"error": "Please describe your symptoms."}), 400

    if len(symptom_text) < 5:
        return jsonify({"error": "Please provide a bit more detail about your symptoms."}), 400

    result = engine.assess(symptom_text)
    return jsonify(result)


@app.route("/scan-medicine", methods=["POST"])
def scan_medicine():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a photo of the medicine strip or prescription."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if file.mimetype not in ALLOWED_IMAGE_TYPES:
        return jsonify({"error": "Please upload a JPEG, PNG, or WEBP image."}), 400

    file.seek(0, 2)  # seek to end to check size
    size = file.tell()
    file.seek(0)
    if size > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image is too large. Please upload a photo under 8MB."}), 400

    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Couldn't read that image. Please try a different photo."}), 400

    result = scanner.scan(image)
    return jsonify(result)


@app.route("/search-medicine", methods=["POST"])
def search_medicine():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()

    if not name:
        return jsonify({"error": "Please enter a medicine name."}), 400

    if len(name) < 3:
        return jsonify({"error": "Please enter at least 3 characters."}), 400

    result = scanner.search_by_name(name)
    return jsonify(result)


if __name__ == "__main__":
    # debug=True only for local development — turn off before deploying
    app.run(debug=True, port=5000)