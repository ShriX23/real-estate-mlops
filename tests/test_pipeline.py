from src.generate_data import generate_dataset
from src.train import FEATURES

def test_generated_data_has_expected_columns_and_rows():
    df = generate_dataset(n_rows=150)
    assert len(df) == 150
    assert set(FEATURES + ["price_inr"]).issubset(df.columns)
    assert (df["price_inr"] > 0).all()

def test_generated_features_are_valid():
    df = generate_dataset(n_rows=150)
    assert (df["area_sqft"] > 0).all()
    assert df["bedrooms"].between(1, 6).all()
    assert df["location_score"].between(1, 10).all()
