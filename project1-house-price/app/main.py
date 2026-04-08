from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import os

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using Random Forest (Ames Dataset)",
    version="2.0.0"
)

# ----------------------------
# Enable CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Path Setup (FIXED FOR RENDER)
# ----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler.pkl")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# ----------------------------
# Load Model
# ----------------------------
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ----------------------------
# Input Schema
# ----------------------------
class HouseInput(BaseModel):
    area: float = Field(..., gt=0)
    quality: int = Field(..., ge=1, le=10)
    garage: int = Field(..., ge=0)
    basement: float = Field(..., ge=0)
    bathroom: int = Field(..., ge=0)
    year_built: int = Field(..., ge=1800, le=2025)
    condition: int = Field(..., ge=1, le=10)

# ----------------------------
# Serve Frontend (ROOT UI)
# ----------------------------
@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ----------------------------
# Static Files (CSS, JS)
# ----------------------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict(data: HouseInput):
    try:
        input_data = np.array([[ 
            data.area,
            data.quality,
            data.garage,
            data.basement,
            data.bathroom,
            data.year_built,
            data.condition
        ]])

        input_scaled = scaler.transform(input_data)

        prediction_usd = model.predict(input_scaled)[0]

        # Convert USD → INR
        prediction_inr = prediction_usd * 83

        return {
            "status": "success",
            "predicted_price": round(prediction_inr, 2),
            "currency": "INR"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))