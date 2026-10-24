from src.generate_data import main as generate_data
from src.train import train, MODEL_PATH, METRICS_PATH

def test_training_produces_model_and_passing_metrics():
    generate_data()
    metrics = train()
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    assert metrics["passed"] is True
    assert metrics["mae_inr"] <= metrics["max_mae_inr"]
