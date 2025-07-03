from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

app = FastAPI()

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="iamaniketmondal/co2-emission-model",
    filename="forecasting_co2_emmision.pkl",
    repo_type="model"
)

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Input schema
class InputData(BaseModel):
    features: list  # Example: [gdp, energy_use, population, ...]

@app.get("/")
def read_root():
    return {"message": "CO2 Emission Prediction API is live!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        prediction = model.predict(arr)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
