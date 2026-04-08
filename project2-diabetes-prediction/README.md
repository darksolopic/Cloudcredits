# 🩺 Diabetes Prediction App

A Machine Learning web application that predicts whether a patient is diabetic based on medical attributes using Logistic Regression.

---

## 🚀 Live Features

* 📊 Predict diabetes based on input data
* ⚡ FastAPI backend
* 🎯 Clean and simple UI
* 🤖 Machine Learning model integration
* 🌐 Ready for deployment (Render)

---

## 📂 Project Structure

```
project2-diabetes-prediction/
│
├── app/
│   ├── main.py        # FastAPI backend
│   └── model.py       # Model training script
│
├── data/
│   └── diabetes.csv   # Dataset
│
├── model/
│   └── diabetes_model.pkl   # Trained model
│
├── templates/
│   └── index.html     # UI
│
├── static/
│   └── style.css      # Styling
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Pima Indians Diabetes Dataset
* Features include:

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age

---

## ⚙️ Model

* Algorithm: **Logistic Regression**
* Preprocessing:

  * Replaced invalid zero values with mean
* Output:

  * **Diabetic**
  * **Not Diabetic**

---

## 🖥️ Run Locally

### 1. Clone repo

```
git clone https://github.com/darksolopic/Cloudcredits.git
cd Cloudcredits/project2-diabetes-prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train model (only once)

```
python app/model.py
```

### 4. Run application

```
python app/main.py
```

### 5. Open browser

```
http://127.0.0.1:8000
```

---

## 🌐 Deployment (Render)

* Root Directory:

```
project2-diabetes-prediction
```

* Build Command:

```
pip install -r requirements.txt
```

* Start Command:

```
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

---

## Future Improvements

* 📈 Show prediction probability
* 🎨 Enhanced UI (React / Tailwind)
* 🔐 User authentication
* 📊 Model comparison (KNN, Random Forest)

---

## 👨‍💻 Author

**Gautam Sinha*

* Data Science | ML | Automation
* Python | FastAPI | Deep Learning

---

## If you like this project

Give it a star ⭐ on GitHub!
