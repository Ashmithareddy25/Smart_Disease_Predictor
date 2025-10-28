# 🩺 Smart Disease Predictor  
**AI-powered Multi-Disease Health Diagnostic Web Application**

---

## 🔍 Overview
**Smart Disease Predictor** is an AI-driven healthcare analytics app built using **Machine Learning (ML)** and **Deep Learning (DL)** models.  
It predicts the likelihood of multiple diseases based on patient data and medical images, serving as an intelligent early diagnostic tool for research and learning purposes.

This project unifies seven disease modules into one simple and interactive Streamlit interface:
- 🩸 Diabetes  
- ❤️ Heart Disease  
- 🧫 Kidney Disease  
- 🧬 Liver Disease  
- 🎗️ Breast Cancer  
- 🦠 Malaria (Image-based)  
- 🫁 Pneumonia (Image-based)

---

## 💡 Features
- Multi-disease prediction using ML & DL models  
- Interactive and visually appealing **Streamlit** UI  
- Handles both tabular data and medical image inputs  
- Fast, lightweight, and easy to deploy  
- Works locally or on **Streamlit Cloud**

---

## ⚙️ Tech Stack
| Component | Technology Used |
|------------|----------------|
| **Frontend** | Streamlit |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | TensorFlow / Keras |
| **Languages** | Python |
| **Libraries** | Pandas, NumPy, TensorFlow, Scikit-learn, Pillow |
| **Deployment** | Streamlit Cloud or Localhost |

---

## 📂 Project Structure
```bash
Smart-Disease-Predictor/
│
├── app_streamlit.py # Main Streamlit app file
├── models/ # ML & DL trained models
│ ├── diabetes.pkl
│ ├── heart.pkl
│ ├── kidney.pkl
│ ├── liver.pkl
│ ├── breast_cancer.pkl
│ ├── malaria.h5
│ └── pneumonia.h5
├── requirements.txt # Dependencies
└── README.md # Project documentation
```


---

## 🧠 How It Works
1. Select a disease module from the sidebar.  
2. For ML models — enter numeric medical data.  
3. For DL models — upload the medical image (e.g., cell or X-ray image).  
4. Click **Predict** to get the result instantly.  
5. The output displays a clear message showing normal or abnormal conditions.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
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

