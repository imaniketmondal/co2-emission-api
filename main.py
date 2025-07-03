from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import requests
import os
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define direct model download URL from Hugging Face
MODEL_URL = "https://huggingface.co/iamaniketmondal/co2-emission-model/resolve/main/forecasting_co2_emmision.pkl"
MODEL_PATH = "model.pkl"

# Download the model if not already downloaded
model = None
try:
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model from Hugging Face via direct URL...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        logging.info("Model downloaded successfully.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Define input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "✅ CO2 Emission Prediction API is running."}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model failed to load. Please check logs or model source."}

    try:
        input_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}
