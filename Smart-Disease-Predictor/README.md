
# 🩺 Smart Disease Predictor  
**AI-powered Multi-Disease Health Diagnostic Web Application**

---

## 🔍 Overview
**Smart Disease Predictor** is an AI-driven healthcare analytics web app built using **Machine Learning (ML)** and **Deep Learning (DL)** models.  
It predicts the likelihood of multiple diseases based on clinical data and medical images, serving as an intelligent early diagnostic tool for research and educational use.

The app offers predictions for seven diseases — Diabetes, Heart Disease, Kidney Disease, Liver Disease, Breast Cancer, Malaria, and Pneumonia — all within one unified, easy-to-use interface.

---

## 💡 Features
- 🩸 **Diabetes Prediction** – Based on medical factors like glucose level, BMI, and insulin.  
- ❤️ **Heart Disease Prediction** – Analyzes cardiac-related parameters to assess heart condition.  
- 🧫 **Kidney Disease Prediction** – Evaluates parameters like serum creatinine and hemoglobin.  
- 🧬 **Liver Disease Prediction** – Checks liver enzymes, bilirubin, and protein ratios.  
- 🎗️ **Breast Cancer Prediction** – Detects malignancy probability using tumor cell features.  
- 🦠 **Malaria Detection (Image)** – Uses a CNN to identify parasitized blood cells.  
- 🫁 **Pneumonia Detection (Image)** – Classifies chest X-ray images as Normal or Pneumonia.

---

## ⚙️ Tech Stack
| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit (for web interface) |
| **Backend ML Models** | Scikit-learn (RandomForest, Logistic Regression, etc.) |
| **Backend DL Models** | TensorFlow / Keras CNN |
| **Language** | Python |
| **Libraries Used** | NumPy, Pandas, TensorFlow, Scikit-learn, Pillow |
| **Deployment** | Streamlit Cloud / Localhost |

---

## 📂 Project Structure
```bash

Smart-Disease-Predictor/
│
├── app_streamlit.py # Main Streamlit application
├── models/ # Trained models (.pkl & .h5 files)
│ ├── diabetes.pkl
│ ├── heart.pkl
│ ├── kidney.pkl
│ ├── liver.pkl
│ ├── breast_cancer.pkl
│ ├── malaria.h5
│ └── pneumonia.h5
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── assets/ # Screenshots and UI images
```


---

## 🧠 How It Works
1. Users select a disease module from the sidebar.  
2. For ML-based diseases (like Diabetes, Heart, etc.), they input numeric medical parameters.  
3. For DL-based diseases (Malaria & Pneumonia), they upload images.  
4. The corresponding trained model predicts the disease outcome.  
5. Results are displayed interactively with clear interpretations.

---

## 🚀 Highlights
- One-stop platform for **multi-disease prediction**
- Combines both **Machine Learning and Deep Learning**
- Simple, interactive, and user-friendly Streamlit UI
- Educational and research-focused healthcare tool
- Works seamlessly across all major devices and browsers

---

## 🧩 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Smart-Disease-Predictor.git
cd Smart-Disease-Predictor
```
# 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

# 3️⃣ Run the application
```
streamlit run app_streamlit.py
```


