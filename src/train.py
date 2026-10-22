
import mlflow
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Chargement des données
data = fetch_california_housing(as_frame=True)

X = data.frame.drop('MedHouseVal', axis=1)
y = data.frame['MedHouseVal']

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Suivi de l'expérience
mlflow.set_experiment("Prediction_Immo_Californie")

with mlflow.start_run():

    # Entraînement
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Évaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Enregistrement MLflow
    mlflow.log_metric("MAE", mae)

    # Sauvegarde du modèle
    joblib.dump(model, "model.pkl")

    print(f"MAE obtenue : {mae:.4f}")
