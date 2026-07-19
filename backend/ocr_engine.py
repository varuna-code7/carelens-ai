"""
Medicine Scanner Engine
-------------------------
OCR extracts text from a photo of a medicine strip/prescription, then
fuzzy-matches the (often messy/misspelled) extracted text against a known
medicine reference table. Same "match noisy text against known patterns"
idea as the symptom engine, applied to a different problem.
"""

import re
import difflib
import pandas as pd
import pytesseract
from PIL import Image, ImageOps, ImageFilter

DATA_PATH = "data/medicine_dataset.csv"
MATCH_THRESHOLD = 0.72  # below this, we don't claim a confident match


class MedicineScanner:
    def __init__(self, data_path: str = DATA_PATH):
        self.df = pd.read_csv(data_path)
        self.variants = []  # list of (variant_text, row_index)
        for idx, row in self.df.iterrows():
            self.variants.append((row["name"].lower(), idx))
            if isinstance(row["aliases"], str):
                for alias in row["aliases"].split("|"):
                    self.variants.append((alias.strip().lower(), idx))

    def extract_text(self, image: Image.Image) -> str:
        # Phone photos carry EXIF orientation — without correcting for it,
        # a sideways photo silently tanks OCR accuracy.
        image = ImageOps.exif_transpose(image)

        if image.width < 800:
            scale = 800 / image.width
            # LANCZOS preserves edge sharpness when upscaling — the previous
            # default resample method softened text edges into a blur,
            # which is what was causing clear photos to fail OCR.
            image = image.resize(
                (800, int(image.height * scale)), resample=Image.Resampling.LANCZOS
            )

        image = image.convert("L")  # grayscale improves OCR on printed text
        image = ImageOps.autocontrast(image)  # boosts text-vs-background contrast
        image = image.filter(ImageFilter.SHARPEN)  # counteracts any softness from resizing

        self._last_processed_image = image  # reused for confidence scoring
        return pytesseract.image_to_string(image)

    def _low_confidence_ratio(self, image: Image.Image) -> float:
        """Fraction of OCR'd words Tesseract itself scored as low-confidence.
        Handwriting mixed with printed letterhead (like a clinic header on a
        prescription) can still average out to a moderate score, but the
        fraction of genuinely low-confidence words stays high — that's the
        more reliable signal for 'this is handwriting, not print'."""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return 1.0

        confidences = [int(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        if not confidences:
            return 1.0
        low_count = sum(1 for c in confidences if c < 50)
        return low_count / len(confidences)

    def _candidate_tokens(self, raw_text: str):
        tokens = re.findall(r"[A-Za-z]{4,}", raw_text)
        return [t.lower() for t in tokens]

    def match(self, raw_text: str):
        tokens = self._candidate_tokens(raw_text)
        if not tokens:
            return None

        best_score = 0.0
        best_row = None
        best_token = None

        for token in tokens:
            for variant_text, row_idx in self.variants:
                score = difflib.SequenceMatcher(None, token, variant_text).ratio()
                if score > best_score:
                    best_score = score
                    best_row = row_idx
                    best_token = token

        if best_row is None or best_score < MATCH_THRESHOLD:
            return None

        row = self.df.iloc[best_row]
        return {
            "matched_name": row["name"],
            "matched_from_text": best_token,
            "confidence": round(best_score * 100, 1),
            "use": row["use"],
            "precautions": row["precautions"],
        }

    def search_by_name(self, typed_text: str):
        """Look up a medicine directly from typed text — same matching
        engine as the scanner, but skips OCR (and the handwriting problem)
        entirely since there's no image involved."""
        result = self.match(typed_text)

        if result is None:
            return {
                "found": False,
                "message": (
                    "We couldn't find that medicine in our database. "
                    "Double-check the spelling, or consult a pharmacist."
                ),
            }

        return {
            "found": True,
            "medicine": result["matched_name"],
            "confidence": result["confidence"],
            "use": result["use"],
            "precautions": result["precautions"],
        }

    def scan(self, image: Image.Image):
        raw_text = self.extract_text(image)
        result = self.match(raw_text)

        if result is None:
            low_conf_ratio = self._low_confidence_ratio(self._last_processed_image)

            if low_conf_ratio > 0.25:
                message = (
                    "We couldn't identify the medicine. This image appears to contain "
                    "handwritten text or unclear printing. CareLens currently works best "
                    "with printed medicine strips, boxes, or labels. Please upload a "
                    "clearer, well-lit image."
                )
            else:
                message = (
                    "Couldn't confidently identify a medicine from this image. "
                    "Try a clearer, well-lit photo, or consult a pharmacist."
                )

            return {
                "found": False,
                "raw_text": raw_text.strip(),
                "low_confidence_ratio": round(low_conf_ratio, 2),
                "message": message,
            }

        return {
            "found": True,
            "raw_text": raw_text.strip(),
            "medicine": result["matched_name"],
            "confidence": result["confidence"],
            "use": result["use"],
            "precautions": result["precautions"],
        }


if __name__ == "__main__":
    from PIL import ImageDraw

    test_words = ["Dolo 650", "Azithral 500", "Ranitidine SR", "XyzUnknownMed 100"]
    scanner = MedicineScanner()

    for word in test_words:
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 35), word, fill="black")
        result = scanner.scan(img)
        print("=" * 60)
        print(f"IMAGE TEXT: {word}")
        print(f"OCR READ:   {result['raw_text']!r}")
        if result["found"]:
            print(f"MATCHED:    {result['medicine']} ({result['confidence']}%)")
            print(f"USE:        {result['use']}")
        else:
            print(f"NO MATCH:   {result['message']}")