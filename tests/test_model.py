
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# Chemin du modèle sauvegardé
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_type():
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestRegressor)


def test_number_of_trees():
    model = joblib.load(MODEL_PATH)
    assert model.n_estimators == 100


def test_model_trained():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "estimators_")
