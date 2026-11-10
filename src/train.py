from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "indoor_position_model.joblib"

MISSING_SIGNAL_VALUE = -110


def prepare_wifi_features(dataframe, wap_columns=None):
    """Select Wi-Fi columns and replace 100 (not detected) with -110."""
    if wap_columns is None:
        wap_columns = [
            column for column in dataframe.columns
            if column.startswith("WAP")
        ]

    features = dataframe[wap_columns].replace(100, MISSING_SIGNAL_VALUE)
    return features, wap_columns


def main():
    train_df = pd.read_csv(DATA_DIR / "trainingData.csv")
    validation_df = pd.read_csv(DATA_DIR / "validationData.csv")

    X_train, wap_columns = prepare_wifi_features(train_df)
    X_validation, _ = prepare_wifi_features(validation_df, wap_columns)

    y_train = train_df[["LONGITUDE", "LATITUDE"]]
    y_validation = validation_df[["LONGITUDE", "LATITUDE"]]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    coordinate_mae = mean_absolute_error(
        y_validation,
        predictions,
        multioutput="raw_values"
    )

    position_errors = np.sqrt(
        ((y_validation.to_numpy() - predictions) ** 2).sum(axis=1)
    )

    r2_values = r2_score(
        y_validation,
        predictions,
        multioutput="raw_values"
    )

    print(f"Longitude MAE: {coordinate_mae[0]:.2f}")
    print(f"Latitude MAE: {coordinate_mae[1]:.2f}")
    print(f"Mean position error: {position_errors.mean():.2f}")
    print(f"Longitude R²: {r2_values[0]:.4f}")
    print(f"Latitude R²: {r2_values[1]:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)

    artifact = {
        "model": model,
        "wap_columns": wap_columns,
        "missing_signal_value": MISSING_SIGNAL_VALUE
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nSaved model artifact to: {MODEL_PATH}")


if __name__ == "__main__":
    main()