from huggingface_hub import hf_hub_download
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Download from Hugging Face Hub
model_path = hf_hub_download(repo_id="your-username/co2-emission-model", filename="model.pkl")

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    y = model.predict(X)
    return {"prediction": y.tolist()}
