from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
import logging

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download and load the model
model = None
try:
    logging.info("Downloading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="iamaniketmondal/co2-emission-model",
        filename="forecasting_co2_emmision.pkl",
        repo_type="model"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

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
