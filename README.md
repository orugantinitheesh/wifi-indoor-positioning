# Wi-Fi-Based Indoor Positioning

This project predicts indoor location coordinates from Wi-Fi signal fingerprints using the UJIIndoorLoc dataset.

## Problem

GPS is unreliable indoors. This project uses Wi-Fi access-point signal strengths collected by a mobile device to predict an indoor position:

- `LONGITUDE`
- `LATITUDE`

The final model is exposed through a FastAPI web service and packaged with Docker.

## Dataset

Source: [UJIIndoorLoc — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)

The dataset contains Wi-Fi fingerprints from three university buildings.

- Training data: 19,937 rows
- Validation data: 1,111 rows
- Wi-Fi input features: 520 (`WAP001` to `WAP520`)
- Targets: `LONGITUDE` and `LATITUDE`

A Wi-Fi value of `100` means the access point was not detected.

### Data retrieval

The raw CSV files are not stored in this repository.

1. Download `trainingData.csv` and `validationData.csv` from the [UJIIndoorLoc dataset page](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).
2. Create a `data/` folder in the project root.
3. Place both files in that folder:

```text
data/
├── trainingData.csv
└── validationData.csv
```

Run `python src/train.py` after downloading the data. It creates the saved model artifact required by the API and Docker image.

## Data preparation

The model uses only the 520 Wi-Fi signal columns.

- Columns not starting with `WAP` are excluded from model input.
- A signal value of `100` is replaced with `-110`, a value weaker than the weakest observed signal (`-104`).
- The raw CSV files are never modified.


## Models compared

Three regression approaches were evaluated with 3-fold cross-validation on the training data.

| Model | Best CV MAE | Validation mean position error |
|---|---:|---:|
| Ridge Regression | 16.42 | 49.80 |
| Tuned K-Nearest Neighbors | 2.26 | 14.37 |
| Tuned Random Forest | 2.88 | 12.45 |

Although K-Nearest Neighbors had the best random cross-validation MAE, the tuned Random Forest generalized better to the separate validation dataset. Therefore, the tuned Random Forest was selected as the final model.

## Final model performance

Final model: `RandomForestRegressor`

- Longitude MAE: 8.65 coordinate units
- Latitude MAE: 7.48 coordinate units
- Mean position error: 12.45 coordinate units
- Median position error: 7.97 coordinate units
- Longitude R²: 0.9865
- Latitude R²: 0.9703

## Project structure

```text
wifi-indoor-positioning/
├── data/
│   ├── trainingData.csv
│   └── validationData.csv
├── models/
│   └── indoor_position_model.joblib
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_building_classifier.ipynb
│   ├── 04_floor_classifier.ipynb
│   ├── 05_coordinate_regression.ipynb
│   └── 06_model_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the final model and save it to `models/`:

```bash
python src/train.py
```

Start the API:

```bash
python -m uvicorn src.serve:app --host 127.0.0.1 --port 9696
```

Check that the service is running:

```bash
curl http://127.0.0.1:9696/health
```

Expected response:

```json
{"status":"ok"}
```

## Prediction API

`POST /predict`

The API accepts a JSON object containing detected Wi-Fi signals. Any omitted WAP is handled as not detected.

```bash
curl -X POST "http://127.0.0.1:9696/predict" \
  -H "Content-Type: application/json" \
  -d "{\"wifi_signals\":{\"WAP001\":-70,\"WAP002\":-82}}"
```

Example response:

```json
{
  "longitude": -7441.15,
  "latitude": 4864859.35
}
```

## Run with Docker

Build the image:

```bash
docker build -t wifi-indoor-positioning .
```

Run the container:

```bash
docker run --rm -p 9696:9696 wifi-indoor-positioning
```

Then use the same `/health` and `/predict` API requests at `http://127.0.0.1:9696`.

## Limitations and next steps

- The coordinates are dataset-specific projected coordinate values, not ordinary GPS latitude/longitude.
- Model performance may change in a new building, with new phones, or if Wi-Fi infrastructure changes.
- Future work could test user-aware validation, add more realistic live Wi-Fi input collection, and deploy the Docker image to a cloud service.