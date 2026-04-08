# 🏠 House Price Prediction App

A full-stack Machine Learning application that predicts house prices based on property features using the Ames Housing Dataset.

---

## 🚀 Live Demo


---

## 📌 Project Overview

This project uses a **Random Forest Regression model** to predict house prices based on key features like area, quality, garage capacity, basement size, and more.

It includes:
- 🔹 Machine Learning model
- 🔹 FastAPI backend
- 🔹 Interactive frontend UI
- 🔹 End-to-end deployment

---

## 🧠 Features

- Predict house prices in real-time
- Clean and user-friendly UI
- REST API built with FastAPI
- Uses advanced ML model (Random Forest)
- Currency conversion (USD → INR)
- Structured project (modular design)

---

## 🏗️ Tech Stack

### 🔹 Backend
- Python
- FastAPI
- NumPy
- Scikit-learn

### 🔹 Frontend
- HTML
- CSS
- JavaScript

### 🔹 Machine Learning
- Random Forest Regressor
- StandardScaler
- Ames Housing Dataset

---

##  Model Details

- **Algorithm:** Random Forest Regressor
- **Features Used:**
  - Living Area
  - Overall Quality
  - Garage Capacity
  - Basement Area
  - Number of Bathrooms
  - Year Built
  - Overall Condition

- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - R² Score

---

## 📁 Project Structure
Project1- House-price/
│
├── app/
│ ├── main.py
│ ├── model.pkl
│ ├── scaler.pkl
│
├── frontend/
│ ├── index.html
│ ├── style.css
│
├── model/
│ ├── train.py
│
├── data/
│ └── AmesHousing.csv
│
├── requirements.txt
└── README.md



## ▶️ Run Locally

### 1️⃣ Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/Cloudcredits.git
cd Cloudcredits/Project1- House-price
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Train model (optional)
cd model
python train.py
4️⃣ Run application
cd ../app
uvicorn main:app --reload
🌐 Access App

Open browser:

http://127.0.0.1:8000