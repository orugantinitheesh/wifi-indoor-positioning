from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "indoor_position_model.joblib"


def load_artifact():
    """Load the saved model and its feature metadata."""
    return joblib.load(MODEL_PATH)


def predict_position(wifi_signals):
    """
    Predict indoor longitude and latitude.

    wifi_signals should be a dictionary, for example:
    {"WAP001": -70, "WAP042": -85}
    """
    artifact = load_artifact()

    model = artifact["model"]
    wap_columns = artifact["wap_columns"]
    missing_signal_value = artifact["missing_signal_value"]

    unknown_columns = set(wifi_signals) - set(wap_columns)

    if unknown_columns:
        raise ValueError(f"Unknown Wi-Fi feature names: {sorted(unknown_columns)}")

    # All access points are initially treated as not detected (100).
    feature_values = {column: 100 for column in wap_columns}

    # Replace only the access points detected by the phone.
    feature_values.update(wifi_signals)

    features = pd.DataFrame([feature_values], columns=wap_columns)
    features = features.replace(100, missing_signal_value)

    prediction = model.predict(features)[0]

    return {
        "longitude": float(prediction[0]),
        "latitude": float(prediction[1])
    }


if __name__ == "__main__":
    # Simple local test using one known validation scan.
    validation_path = PROJECT_ROOT / "data" / "validationData.csv"
    validation_df = pd.read_csv(validation_path)

    artifact = load_artifact()
    wap_columns = artifact["wap_columns"]

    sample_signals = validation_df.loc[0, wap_columns].to_dict()

    result = predict_position(sample_signals)

    print("Predicted position:", result)
    print(
        "Actual position:",
        {
            "longitude": float(validation_df.loc[0, "LONGITUDE"]),
            "latitude": float(validation_df.loc[0, "LATITUDE"])
        }
    )