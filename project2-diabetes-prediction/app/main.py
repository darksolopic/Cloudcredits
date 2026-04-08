from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import webbrowser
import os

app = FastAPI()

# ✅ BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "model", "diabetes_model.pkl")

# ✅ Static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ✅ Function to load HTML
def load_html(result=""):
    with open(TEMPLATE_PATH, "r") as f:
        html = f.read()
    return html.replace("{{ result }}", result)


# ✅ Home route
@app.get("/", response_class=HTMLResponse)
def home():
    return load_html()


# ✅ Predict route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree: float = Form(...),
    age: float = Form(...)
):
    try:
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])

        prediction = model.predict(data)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

    except Exception as e:
        result = f"Error: {str(e)}"

    return load_html(result)


# ✅ Run locally
if __name__ == "__main__":
    import uvicorn
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)