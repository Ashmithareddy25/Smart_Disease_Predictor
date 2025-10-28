import streamlit as st
import numpy as np
import pickle, os
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
        body {background: linear-gradient(135deg,#f3e7ff,#e3f0ff);}
        h1,h2,h3,h4,h5 {color:#4e0374;font-family:Cambria;}
        .stButton>button {
            background-color:#4e0374;color:white;border-radius:8px;
            font-size:16px;padding:0.6em 1.2em;
        }
        .stButton>button:hover {background-color:#c37ee6;color:black;}
        .info {
            background-color:#f6f1ff;border-left:5px solid #4e0374;
            padding:10px;margin-top:10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Smart Disease Predictor")
st.caption("AI-powered multi-disease health screening using Machine Learning and Deep Learning models")

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def load_model(path):
    """Safely load sklearn (.pkl) or keras (.h5) model."""
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".h5"):
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None


def safe_predict(model, features):
    """Predict safely, automatically padding/trimming to match model expectations."""
    try:
        arr = np.array([features])
        expected = getattr(model, "n_features_in_", arr.shape[1])
        current = arr.shape[1]

        # Adjust shape without displaying warnings
        if current < expected:
            diff = expected - current
            arr = np.hstack([arr, np.zeros((1, diff))])
        elif current > expected:
            arr = arr[:, :expected]

        return model.predict(arr)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.header("Select Disease Module")
page = st.sidebar.radio("Go to:",
    ["🏠 Home","🩸 Diabetes","❤️ Heart Disease",
     "🧫 Kidney Disease","🧬 Liver Disease",
     "🎗️ Breast Cancer","🦠 Malaria (Image)","🫁 Pneumonia (Image)"]
)

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.subheader("Welcome to Smart Disease Predictor")
    st.write("""
        This application predicts the likelihood of several medical conditions
        based on patient data or medical images.
        It integrates classical Machine Learning models (.pkl) and Deep Learning CNN models (.h5).
    """)
    st.markdown("""
        ### Supported Diseases:
        - 🩸 **Diabetes** — PIMA Indians dataset  
        - ❤️ **Heart Disease** — Cleveland Heart dataset  
        - 🧫 **Kidney Disease** — Chronic Kidney dataset  
        - 🧬 **Liver Disease** — Indian Liver dataset  
        - 🎗️ **Breast Cancer** — Wisconsin Diagnostic dataset  
        - 🦠 **Malaria / 🫁 Pneumonia** — Image-based CNN models  
    """)
    st.info("Use the sidebar to select a disease. Fill in all details before clicking *Predict*.")

# ─────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────
elif page == "🩸 Diabetes":
    st.header("🩸 Diabetes Prediction")
    st.markdown('<div class="info">Normal fasting glucose: 70–99 mg/dL • BMI: 18.5–24.9 normal</div>', unsafe_allow_html=True)

    fields = {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Plasma Glucose Concentration (mg/dL)",
        "BloodPressure": "Diastolic Blood Pressure (mm Hg)",
        "SkinThickness": "Triceps Skin-Fold Thickness (mm)",
        "Insulin": "2-Hour Serum Insulin (µU/mL)",
        "BMI": "Body Mass Index (kg/m²)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function (Family History)",
        "Age": "Age (in years)"
    }

    inputs = [st.number_input(v, 0.0, 500.0, 0.0) for v in fields.values()]

    if st.button("Predict Diabetes"):
        mdl = load_model("models/diabetes.pkl")
        if mdl:
            pred = safe_predict(mdl, inputs)
            if pred == 0:
                st.success("✅ Negative – No signs of diabetes.")
            else:
                st.error("⚠️ Positive – High likelihood of diabetes. Consult a healthcare provider.")

# ─────────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────────
elif page == "❤️ Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    st.markdown('<div class="info">Normal BP < 120/80 mmHg • Cholesterol < 200 mg/dL • Max Heart Rate > 100 bpm</div>', unsafe_allow_html=True)

    fields = {
        "age": "Age (in years)",
        "sex": "Sex (1 = Male, 0 = Female)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Serum Cholesterol (mg/dL)",
        "fbs": "Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)",
        "restecg": "Resting Electrocardiographic Results (0–2)",
        "thalach": "Maximum Heart Rate Achieved",
        "exang": "Exercise-Induced Angina (1 = Yes, 0 = No)",
        "oldpeak": "ST Depression Induced by Exercise",
        "slope": "Slope of Peak Exercise ST Segment (0–2)",
        "ca": "Number of Major Vessels Colored by Fluoroscopy (0–3)",
        "thal": "Thalassemia Type (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)"
    }

    inputs = [st.number_input(v, 0.0, 500.0, 0.0) for v in fields.values()]

    if st.button("Predict Heart Disease"):
        mdl = load_model("models/heart.pkl")
        if mdl:
            pred = safe_predict(mdl, inputs)
            if pred == 0:
                st.success("✅ Negative – The heart appears healthy and normal.")
            else:
                st.error("⚠️ Positive – Possible signs of heart disease detected.")

# ─────────────────────────────────────────────
# KIDNEY
# ─────────────────────────────────────────────
elif page == "🧫 Kidney Disease":
    st.header("🧫 Kidney Disease Prediction")
    st.markdown('<div class="info">Normal Creatinine: 0.6–1.3 mg/dL • Hemoglobin: 12–16 g/dL</div>', unsafe_allow_html=True)

    fields = {
        "age": "Age (in years)",
        "blood_pressure": "Blood Pressure (mm Hg)",
        "specific_gravity": "Specific Gravity (1.005 – 1.025)",
        "albumin": "Albumin (g/dL)",
        "sugar": "Sugar (mg/dL)",
        "blood_glucose_random": "Random Blood Glucose (mg/dL)",
        "blood_urea": "Blood Urea (mg/dL)",
        "serum_creatinine": "Serum Creatinine (mg/dL)",
        "sodium": "Sodium (mEq/L)",
        "potassium": "Potassium (mEq/L)",
        "haemoglobin": "Hemoglobin (g/dL)",
        "packed_cell_volume": "Packed Cell Volume (%)",
        "white_blood_cell_count": "White Blood Cell Count (/mm³)",
        "red_blood_cell_count": "Red Blood Cell Count (millions/µL)"
    }

    inputs = [st.number_input(v, 0.0, 1000.0, 0.0) for v in fields.values()]

    if st.button("Predict Kidney Disease"):
        mdl = load_model("models/kidney.pkl")
        if mdl:
            pred = safe_predict(mdl, inputs)
            if pred == 0:
                st.success("✅ Negative – Kidney function appears normal.")
            else:
                st.error("⚠️ Positive – Possible kidney abnormality detected.")

# ─────────────────────────────────────────────
# LIVER
# ─────────────────────────────────────────────
elif page == "🧬 Liver Disease":
    st.header("🧬 Liver Disease Prediction")
    st.markdown('<div class="info">Normal Bilirubin < 1.2 mg/dL • Albumin/Globulin Ratio: 1.0 – 2.0</div>', unsafe_allow_html=True)

    fields = {
        "Total_Bilirubin": "Total Bilirubin (mg/dL)",
        "Direct_Bilirubin": "Direct Bilirubin (mg/dL)",
        "Alkaline_Phosphotase": "Alkaline Phosphotase (IU/L)",
        "Alamine_Aminotransferase": "Alamine Aminotransferase (ALT – IU/L)",
        "Aspartate_Aminotransferase": "Aspartate Aminotransferase (AST – IU/L)",
        "Total_Protiens": "Total Proteins (g/dL)",
        "Albumin": "Albumin (g/dL)",
        "Albumin_and_Globulin_Ratio": "Albumin and Globulin Ratio"
    }

    inputs = [st.number_input(v, 0.0, 1000.0, 0.0) for v in fields.values()]

    if st.button("Predict Liver Disease"):
        mdl = load_model("models/liver.pkl")
        if mdl:
            pred = safe_predict(mdl, inputs)
            if pred == 0:
                st.success("✅ Negative – Liver function appears normal.")
            else:
                st.error("⚠️ Positive – Possible liver disorder detected.")

# ─────────────────────────────────────────────
# BREAST CANCER
# ─────────────────────────────────────────────
elif page == "🎗️ Breast Cancer":
    st.header("🎗️ Breast Cancer Prediction")
    st.markdown('<div class="info">Regular screening and early diagnosis are crucial for recovery.</div>', unsafe_allow_html=True)

    fields = {
        "radius_mean": "Mean Radius",
        "texture_mean": "Mean Texture",
        "perimeter_mean": "Mean Perimeter",
        "area_mean": "Mean Area",
        "smoothness_mean": "Mean Smoothness",
        "compactness_mean": "Mean Compactness",
        "concavity_mean": "Mean Concavity",
        "concave_points_mean": "Mean Concave Points",
        "symmetry_mean": "Mean Symmetry",
        "fractal_dimension_mean": "Mean Fractal Dimension"
    }

    inputs = [st.number_input(v, 0.0, 1000.0, 0.0) for v in fields.values()]

    if st.button("Predict Breast Cancer"):
        mdl = load_model("models/breast_cancer.pkl")
        if mdl:
            pred = safe_predict(mdl, inputs)
            if pred == 1:
                st.success("✅ Negative – No malignancy detected.")
            else:
                st.error("⚠️ Positive – Possible malignant cells detected.")

# ─────────────────────────────────────────────
# MALARIA (IMAGE)
# ─────────────────────────────────────────────
elif page == "🦠 Malaria (Image)":
    st.header("🦠 Malaria Detection (Image)")
    st.markdown('<div class="info">Upload a microscopic blood-cell image. Model classifies it as *Parasitized* or *Uninfected*.</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload Cell Image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).resize((64,64))
        st.image(img, width=200)
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        mdl = load_model("models/malaria.h5")
        if mdl and st.button("Predict Malaria"):
            pred = mdl.predict(arr)
            label = "⚠️ Positive – Parasitized cell detected." if pred[0][0] > 0.5 else "✅ Negative – Healthy cell."
            st.success(label)

# ─────────────────────────────────────────────
# PNEUMONIA (IMAGE)
# ─────────────────────────────────────────────
elif page == "🫁 Pneumonia (Image)":
    st.header("🫁 Pneumonia Detection (Chest X-ray)")
    st.markdown('<div class="info">Upload a chest X-ray image to detect pneumonia.</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).convert("RGB").resize((150,150))
        st.image(img, width=250)
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        mdl = load_model("models/pneumonia.h5")
        if mdl and st.button("Predict Pneumonia"):
            pred = mdl.predict(arr)
            label = "⚠️ Positive – Pneumonia detected." if pred[0][0] > 0.5 else "✅ Negative – Normal lungs."
            st.success(label)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 Smart Disease Predictor · Developed by Ashmitha Reddy Thota")
