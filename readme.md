 CareLens AI

**An AI-powered health accessibility assistant that helps people understand symptom urgency and medicine information in plain language — not a diagnosis, just clarity on what to do next.**

Built for the Idea2Impact Online Hackathon 2026 · Theme 3: Crisis Management, HealthTech & Emergency Respons


## The problem

Many people — especially in rural areas, elderly populations, and those with limited medical knowledge — don't know if their symptoms are urgent, don't understand medicine labels or prescriptions, and delay seeking care because they lack clear guidance. CareLens helps people make informed decisions with health *guidance*, not diagnoses.

## What it does

CareLens has two focused tools, built on one shared AI approach: matching free-text input against known patterns using **TF-IDF vectorization + cosine similarity**.

### 1. Symptom Checker
- Describe symptoms in your own words (typed or spoken)
- Get a **risk level** (🟢 Low / 🟡 Moderate / 🔴 High) — never a named disease, by design, to avoid false self-diagnosis
- See **why** — the specific words that drove the reading
- Get a plain-language explanation and safe self-care guidance, scoped to the risk level
- Always-visible warning signs that mean "seek care now," regardless of the computed risk
- **Voice input** in English, Tamil, Hindi, Telugu, Malayalam, and Kannada — non-English speech is transcribed and automatically translated to English before analysis, with the original transcript shown for verification
- Recent-checks history, stored locally in the browser only

### 2. Medicine Scanner
- Photograph a printed medicine strip, box, or label — OCR (Tesseract) reads it, then fuzzy-matches the (often messy) extracted text against a medicine reference table, including common Indian brand names (e.g. "Dolo" → Paracetamol)
- Or just type a medicine name directly — same matching engine, no photo needed
- Explains what it's for and key precautions, in plain language
- Detects and clearly flags handwritten/unclear input rather than guessing, using Tesseract's own per-word confidence scores

## Why AI is central, not decorative

Both features run on a real, functional matching pipeline — not a wrapper around a general-purpose chatbot:
- **TF-IDF + cosine similarity** for symptom-to-risk-pattern matching, trained on the [Symptom2Disease dataset](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) (24 condition patterns)
- **OCR (Tesseract) + fuzzy string matching** for medicine identification, with image preprocessing (EXIF correction, upscaling, contrast, sharpening) and confidence-based handwriting detection
- Every result includes the contributing terms/factors — the reasoning is inspectable, not a black box

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML / CSS / JavaScript (no framework, no build step) |
| Backend | Python, Flask, Flask-CORS |
| Symptom AI | scikit-learn (TF-IDF, cosine similarity), pandas |
| Medicine OCR | Tesseract OCR, pytesseract, Pillow |
| Translation | MyMemory Translation API (free tier, no key) |
| Voice input | Browser-native Web Speech API |
| Deployment | Render (backend) · Netlify (frontend) |

## Project structure

```
carelens-ai/
├── backend/
│   ├── app.py                    # Flask app, all API routes
│   ├── triage_engine.py          # Symptom AI: TF-IDF + cosine similarity
│   ├── ocr_engine.py             # Medicine AI: OCR + fuzzy matching
│   ├── requirements.txt
│   └── data/
│       ├── symptom_dataset.csv   # Symptom2Disease dataset
│       └── medicine_dataset.csv  # Medicine reference table
│
├── frontend/
│   ├── home.html                 # Landing page
│   ├── symptom-checker.html      # Symptom checker + voice input
│   ├── medicine-scanner.html     # Medicine scanner + search
│
└── README.md
```

## Running it locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract OCR separately (it's a system binary, not a Python package):
- **Windows:** [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

Then:
```bash
python app.py
```
Runs on `http://127.0.0.1:5000`.

### Frontend

No build step — just open `frontend/home.html` in a browser, or serve the folder with any static server (e.g. VS Code Live Server). Before testing, make sure `API_URL` / `SEARCH_URL` in each HTML file's `<script>` point to your running backend (`http://127.0.0.1:5000` for local dev).

## API reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/` | GET | — | Health check |
| `/assess-symptoms` | POST | `{ "symptoms": "..." }` | Risk level, confidence, contributing factors, explanation, self-care tips, warning signs |
| `/scan-medicine` | POST | `multipart/form-data`, field `image` | Medicine name, confidence, use, precautions (or a clear "not identified" message) |
| `/search-medicine` | POST | `{ "name": "..." }` | Same shape as above, text-only lookup |

## Known limitations (by design, not hidden)

- **Not a diagnosis.** CareLens gives a risk *reading*, never names a specific disease — this is a deliberate safety choice, not a missing feature.
- **Medicine scanner works best on printed text.** Handwritten prescriptions are a genuinely hard, unsolved OCR problem — the app detects this using Tesseract's confidence scores and tells the user clearly, rather than guessing.
- **Voice recognition quality varies by language.** English is most reliable; Indian regional language support depends on the browser's underlying speech engine, which is still maturing.
- **Translation uses a free-tier API** (no key, rate-limited) — fine for demo/personal use, would need a paid tier for production scale.
- **Small reference datasets** (24 symptom categories, 15 medicines) — built for demo scope within the hackathon timeline; designed to be easily extended with a larger dataset.

## Disclaimer

CareLens provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with questions about a medical condition.

---

Built for Idea2Impact Online Hackathon 2026 by _Muthu Varuna_.
