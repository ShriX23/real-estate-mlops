from pathlib import Path
import joblib
import pandas as pd
import bentoml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]

@bentoml.service(name="real_estate_predictor", traffic={"timeout": 10})
class HousingService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model missing. Run `python -m src.train` first.")
        self.model = joblib.load(MODEL_PATH)

    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        missing = [key for key in FEATURES if key not in input_data]
        if missing:
            return {"error": f"Missing fields: {missing}"}
        try:
            row = pd.DataFrame([{key: float(input_data[key]) for key in FEATURES}], columns=FEATURES)
        except (TypeError, ValueError):
            return {"error": "All feature values must be numeric"}
        if row.isnull().any().any() or (row < 0).any().any():
            return {"error": "Feature values must be non-negative numbers"}
        return {"predicted_price_inr": round(float(self.model.predict(row)[0]), 2), "currency": "INR"}
