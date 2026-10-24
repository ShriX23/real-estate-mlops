from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]
MAX_MAE_INR = 650000

def train():
    if not DATA_PATH.exists():
        from src.generate_data import main
        main()
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["price_inr"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if df[required].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    if len(df) < 100:
        raise ValueError("Need at least 100 data rows")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["price_inr"], test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    passed = mae <= MAX_MAE_INR
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {"mae_inr": round(mae, 2), "r2": round(r2, 4),
               "max_mae_inr": MAX_MAE_INR, "passed": passed,
               "rows": int(len(df)), "features": FEATURES}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise RuntimeError(f"Model rejected: MAE {mae:.2f} > {MAX_MAE_INR}")
    return metrics

if __name__ == "__main__":
    train()
