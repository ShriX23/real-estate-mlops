from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"

def generate_dataset(n_rows=1200, seed=42):
    # Synthetic demo data only; not real estate market observations.
    rng = np.random.default_rng(seed)
    area = rng.integers(350, 3200, size=n_rows)
    bedrooms = np.clip(np.round(area / 550 + rng.normal(0, .7, n_rows)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, size=n_rows), 1, 5).astype(int)
    age = rng.integers(0, 35, size=n_rows)
    location_score = rng.integers(1, 11, size=n_rows)
    noise = rng.normal(0, 350000, size=n_rows)
    price = np.maximum(area * 7500 + bedrooms * 350000 + bathrooms * 180000
                       - age * 22000 + location_score * 240000 + noise, 500000).round().astype(int)
    return pd.DataFrame({"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                         "age_years": age, "location_score": location_score, "price_inr": price})

def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Created {len(df)} synthetic rows at {DATA_PATH}")

if __name__ == "__main__":
    main()
