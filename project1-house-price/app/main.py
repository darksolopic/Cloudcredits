from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle

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
# Load Model
# ----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ----------------------------
# Input Schema (UPDATED)
# ----------------------------
class HouseInput(BaseModel):
    area: float = Field(..., gt=0, example=1500)
    quality: int = Field(..., ge=1, le=10, example=7)
    garage: int = Field(..., ge=0, example=2)
    basement: float = Field(..., ge=0, example=800)
    bathroom: int = Field(..., ge=0, example=2)
    year_built: int = Field(..., ge=1800, le=2025, example=2005)
    condition: int = Field(..., ge=1, le=10, example=5)

# ----------------------------
# Serve Frontend (ROOT UI)
# ----------------------------
@app.get("/")
def serve_ui():
    return FileResponse("../frontend/index.html")

# ----------------------------
# Static Files (CSS, JS)
# ----------------------------
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# ----------------------------
# Prediction Endpoint (UPDATED)
# ----------------------------
@app.post("/predict")
def predict(data: HouseInput):
    try:
        # Prepare input (7 features)
        input_data = np.array([[ 
            data.area,
            data.quality,
            data.garage,
            data.basement,
            data.bathroom,
            data.year_built,
            data.condition
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict (USD)
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