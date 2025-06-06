# Badminton Speed Prediction API

This is a FastAPI application that predicts badminton shuttlecock speed based on sensor data using a pre-trained TensorFlow model.

## Project Structure

```
.AIOT-Badminton-Speed-API/
├── app/
│   ├── __init__.py
│   ├── core/             # Core configurations and utilities
│   │   ├── __init__.py
│   │   └── config.py     # Model loading and lifespan management
│   ├── models/           # Data models (Pydantic schemas)
│   │   ├── __init__.py
│   │   └── schemas.py    # Defines SensorFrame, PredictionRequest, etc.
│   ├── services/         # Business logic and helper functions
│   │   ├── __init__.py
│   │   └── prediction_service.py # Data preprocessing and validation
│   └── main.py           # Main FastAPI application
├── ml_model/             # Machine learning models and scalers
│   ├── __init__.py
│   ├── badminton_speed_predictor.h5
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
├── data/                 # Data files (e.g., CSV, JSON)
├── requirements.txt      # Python dependencies
├── api_test.py           # API testing script
├── train.py              # Model training script
└── README.md             # Project README
```

## Setup and Installation

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone <repository_url>
    cd AIOT-Badminton-Speed-API
    ```

2.  **Install `uv` (if you haven't already):**

    ```bash
    pip install uv
    ```

3.  **Create a virtual environment and install dependencies using `uv`:**

    ```bash
    uv sync
    ```

## Running the Application

To start the FastAPI application, run the following command from the project root directory:

```bash
uv run ./app/main.py
```

The API will be accessible at `http://localhost:8000`.

## API Endpoints

*   **Root:** `/`
    *   `GET`: Returns a welcome message and available endpoints.

*   **Health Check:** `/health`
    *   `GET`: Checks the health of the service and model loading status.

*   **Predict Speed:** `/predict_speed`
    *   `POST`: Predicts badminton shuttlecock speed based on provided sensor data.
        *   **Request Body:** A JSON object containing a `sensor_data` field, which is a list of 30 `SensorFrame` objects.
        *   **Response Body:** A JSON object with `predicted_speed` (in km/h) and `confidence_info`.

*   **Predict Speed Batch:** `/predict_speed_batch`
    *   `POST`: Predicts badminton shuttlecock speed for multiple sensor data inputs in a single request.
        *   **Request Body:** A JSON object containing a `sensor_data_batch` field, which is a list of `PredictionRequest` objects.
        *   **Response Body:** A JSON object with a list of `PredictionResponse` objects.

## Testing the API

You can use the `api_test.py` script to test the prediction endpoint. Make sure the FastAPI application is running before executing the test script.

```bash
uv run api_test.py
```

## Model and Data

*   The pre-trained TensorFlow model (`badminton_speed_predictor.h5`) and scikit-learn scalers (`feature_scaler.pkl`, `target_scaler.pkl`) are located in the `ml_model/` directory.
*   Raw and processed data files are stored in the `data/` directory.

## Development

*   **`train.py`**: Script used for training the speed prediction model.
*   **`label_pseudo_speed.py`**: Script for pseudo-labeling data.
*   **`label_real_speed.py`**: Script for real-labeling data.
*   **`correlation.py`**: Script for analyzing data correlations.
*   **`descibe.py`**: Script for describing data.

Feel free to explore the code and contribute!