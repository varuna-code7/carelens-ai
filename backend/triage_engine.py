"""
Symptom Triage Engine
----------------------
Same core technique as the PM Internship Recommender: TF-IDF vectorization +
cosine similarity. Instead of matching resumes to jobs, this matches a
user's free-text symptom description to known disease patterns.

Swap DATA_PATH to the real Kaggle Symptom2Disease CSV once downloaded.
Expected columns: 'label' (disease name), 'text' (symptom description).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/symptom2disease.csv"

# Generic/vague words that are grammatically valid but carry no clinical
# meaning on their own (e.g. "I feel bad" -> "bad" isn't a symptom).
# scikit-learn's built-in English stopword list doesn't cover these since
# they're everyday adjectives/verbs, not function words.
EXTRA_STOPWORDS = {
    "feel", "feeling", "felt", "feels", "bad", "good", "really", "quite",
    "lot", "lots", "ive", "also", "day", "days", "since", "time", "like",
    "just", "getting", "got", "been", "being", "having", "having", "much",
    "very", "sometimes", "often", "recently", "lately",
}
CUSTOM_STOPWORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

# Internal only — never shown to the user. Used to derive a RISK LEVEL,
# not a diagnosis. Refine/expand once real disease names come in from the
# Kaggle dataset. Anything unmapped defaults to "moderate" (safer default
# than assuming low risk).
RISK_MAP = {
    "Common Cold": "low",
    "Skin Allergy": "low",
    "Anxiety": "low",
    "Migraine": "moderate",
    "Food Poisoning": "moderate",
    "Dengue Fever": "moderate",
    "Typhoid": "moderate",
    "Asthma Attack": "high",
    "Chest Pain - Cardiac": "high",
    "Severe Allergic Reaction": "high",
    "Stroke Symptoms": "high",
}

RISK_DISPLAY = {
    "low": {
        "label": "Low Risk",
        "icon": "🟢",
        "recommendation": "Likely manageable at home. Monitor symptoms and rest.",
        "explanation": "Your symptoms match patterns that are usually mild and "
                        "self-limiting — the kind of thing your body typically "
                        "recovers from on its own with rest and basic care.",
        "self_care": [
            "Drink plenty of water throughout the day.",
            "Get extra rest and avoid overexertion.",
            "Eat light, easy-to-digest food.",
            "Keep track of your symptoms — note if anything changes or worsens.",
        ],
    },
    "moderate": {
        "label": "Moderate Risk",
        "icon": "🟡",
        "recommendation": "Consider consulting a healthcare professional, especially if symptoms persist or worsen.",
        "explanation": "Your symptoms match patterns that sometimes resolve on "
                        "their own but can also need medical attention — "
                        "particularly if they last more than a couple of days "
                        "or start getting worse rather than better.",
        "self_care": [
            "Rest and stay well hydrated while you monitor things.",
            "Track how long symptoms have lasted and any changes day to day.",
            "Avoid self-medicating beyond basic, familiar remedies.",
            "Plan to see a doctor if there's no improvement within 2–3 days.",
        ],
    },
    "high": {
        "label": "High Risk",
        "icon": "🔴",
        "recommendation": "Seek medical attention promptly. If severe, treat as an emergency.",
        "explanation": "Your symptoms match patterns that can be serious and "
                        "sometimes escalate quickly. This isn't the moment for "
                        "home remedies — the safest move is getting evaluated "
                        "by a professional.",
        "self_care": [
            "Don't try to wait this out at home — arrange care now.",
            "If possible, have someone stay with you or take you in.",
            "Avoid exertion while you arrange transport or care.",
            "If symptoms are severe or rapidly worsening, treat this as an emergency.",
        ],
    },
}

# Shown regardless of assessed risk — these are the signs that always mean
# "escalate now," independent of what the model thinks the risk level is.
WARNING_SIGNS = [
    "Difficulty breathing",
    "Chest pain or pressure",
    "Sudden confusion or slurred speech",
    "Severe or worsening symptoms",
]


class TriageEngine:
    def __init__(self, data_path: str = DATA_PATH):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOPWORDS)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _contributing_factors(self, query_vec, matched_row_index, top_n=3):
        """Pull the query terms that most drove the match — this is what
        powers 'Why This Recommendation?' without naming a disease."""
        query_terms = query_vec.toarray().flatten()
        doc_terms = self.tfidf_matrix[matched_row_index].toarray().flatten()
        overlap_scores = query_terms * doc_terms  # high where both value the term
        top_indices = overlap_scores.argsort()[::-1][:top_n]
        factors = [self.feature_names[i] for i in top_indices if overlap_scores[i] > 0]
        return factors

    def assess(self, symptom_text: str):
        query_vec = self.vectorizer.transform([symptom_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_index = similarities.argmax()
        best_similarity = float(similarities[best_index])
        matched_disease = self.df.iloc[best_index]["label"]  # internal only

        # Count meaningful (non-stopword) terms the query actually contributed.
        # A short, vague input like "why this bad" can still produce a
        # deceptively high cosine similarity off a single shared word — so
        # we require enough real content, not just a confidence score, before
        # trusting the match.
        meaningful_terms = int((query_vec.toarray() > 0).sum())

        low_information = meaningful_terms < 2
        low_confidence = best_similarity < 0.15

        if low_information or low_confidence:
            risk = "moderate"
            factors = []
            note = ("Your description was quite brief. For a more reliable reading, "
                    "add more detail — where the symptom is, how long it's lasted, "
                    "and how severe it feels.")
        else:
            risk = RISK_MAP.get(matched_disease, "moderate")
            factors = self._contributing_factors(query_vec, best_index)
            note = None

        display = RISK_DISPLAY[risk]
        return {
            "risk_level": display["label"],
            "risk_icon": display["icon"],
            "confidence": round(best_similarity * 100, 1),
            "contributing_factors": factors,
            "recommendation": display["recommendation"],
            "explanation": display["explanation"],
            "self_care": display["self_care"],
            "warning_signs": WARNING_SIGNS,
            "note": note,
        }


if __name__ == "__main__":
    engine = TriageEngine()

    test_cases = [
        "I have a crushing pain in my chest and it's spreading to my left arm, sweating a lot",
        "runny nose and sneezing since this morning, throat is a bit scratchy",
        "sudden difficulty breathing, wheezing, chest feels very tight",
        "itchy red rash on my hand after using new soap",
    ]

    for text in test_cases:
        print("=" * 70)
        print(f"INPUT: {text}")
        r = engine.assess(text)
        print(f"\n{r['risk_icon']} {r['risk_level']}  (confidence: {r['confidence']}%)\n")
        if r["contributing_factors"]:
            print("Based on:")
            for f in r["contributing_factors"]:
                print(f"  • {f}")
        print(f"\nRecommendation:\n  {r['recommendation']}")
        print("\nWarning signs (seek care immediately if present):")
        for w in r["warning_signs"]:
            print(f"  • {w}")
        print()