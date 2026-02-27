import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# =======================================================
# PAGE CONFIG (set once, at the very top)
# =======================================================
st.set_page_config(
    page_title="African Diabetes Predictor",
    page_icon=None,
    layout="wide"
)

# =======================================================
# HELPER FUNCTIONS FOR ASSISTANT
# =======================================================

def classify_question(question: str) -> str:
    """Roughly classify what the user is asking about."""
    q = question.lower()
    if any(word in q for word in ["preg", "pregnancies", "pregnancy"]):
        return "pregnancies_info"
    if any(word in q for word in ["glucose", "sugar level", "blood sugar"]):
        return "glucose_info"
    if any(word in q for word in ["bmi", "weight", "obesity", "overweight"]):
        return "bmi_info"
    if "insulin" in q:
        return "insulin_info"
    if any(word in q for word in ["blood pressure", "bp", "hypertension"]):
        return "bp_info"
    return "general_info"


def short_parameter_explanation(category: str) -> str:
    """Short explanations for specific parameters."""
    if category == "pregnancies_info":
        return "“Preg” refers to the **number of pregnancies** the patient has had."
    if category == "glucose_info":
        return "“Glucose” is the **blood sugar concentration**, usually measured in mg/dL."
    if category == "bmi_info":
        return (
            "BMI (**Body Mass Index**) is a rough measure of body fat based on "
            "height and weight."
        )
    if category == "insulin_info":
        return (
            "“Insulin” here is the **blood insulin level**, which reflects how "
            "the pancreas is working."
        )
    if category == "bp_info":
        return (
            "“BP” is **blood pressure**. Persistently high blood pressure "
            "can increase cardiovascular and diabetes-related risks."
        )
    return ""


def simple_health_reply(question: str) -> str:
    """
    Very simple, non-medical helper that gives general information
    about diabetes risk factors. It is NOT a diagnostic tool.
    """
    q = question.lower()
    parts = []

    # Always start with a disclaimer
    parts.append(
        "This answer is for general information only and is **not** a diagnosis "
        "or medical advice."
    )

    if any(word in q for word in ["glucose", "sugar", "blood sugar"]):
        parts.append(
            "High blood glucose is one of the main risk factors for type 2 "
            "diabetes. Doctors may look at fasting glucose, an oral "
            "glucose tolerance test, or HbA1c to understand longer-term control."
        )
        parts.append(
            "Lifestyle changes like healthier diet, regular physical activity, "
            "and weight management can sometimes improve glucose levels, but "
            "any treatment decisions must be made with a healthcare professional."
        )

    elif any(word in q for word in ["bmi", "weight", "overweight", "obese"]):
        parts.append(
            "BMI (Body Mass Index) is a rough indicator of whether weight is in a "
            "low, normal, or high range for a given height. Higher BMI can be "
            "linked with increased diabetes and cardiovascular risk, but it does "
            "not tell the whole story."
        )
        parts.append(
            "A balanced diet, more movement, and sleep/stress management can help, "
            "but please talk with a professional before making big changes."
        )

    elif any(word in q for word in ["blood pressure", "bp", "hypertension"]):
        parts.append(
            "Raised blood pressure (hypertension) can damage blood vessels over "
            "time and is often seen together with diabetes."
        )
        parts.append(
            "Reducing salt intake, avoiding smoking, moderating alcohol, and "
            "staying active can help, but medication decisions must be made by a "
            "clinician."
        )

    elif "pregnant" in q or "pregnancy" in q:
        parts.append(
            "Diabetes in pregnancy (including gestational diabetes) needs close "
            "monitoring by a healthcare team. Online tools like this cannot "
            "assess pregnancy risk."
        )

    else:
        parts.append(
            "Diabetes risk is usually assessed using a combination of factors: "
            "glucose, blood pressure, weight/BMI, age, family history, and "
            "sometimes cholesterol and other labs."
        )

    # Always end with a strong safety reminder
    parts.append(
        "For any personal health concern, new symptoms, or abnormal lab results, "
        "please contact a doctor or qualified health professional. If something "
        "feels urgent, seek emergency care."
    )

    return "\n\n".join(parts)


def generate_health_answer(question: str) -> str:
    """
    Combined assistant:
    1) Short explanation of specific parameters (if relevant)
    2) General education about diabetes risk factors
    """
    category = classify_question(question)
    short_part = short_parameter_explanation(category)
    long_part = simple_health_reply(question)

    if short_part:
        return f"{short_part}\n\n---\n\n{long_part}"
    else:
        return long_part


# ---------------------------------------------------------
# SAFE INITIALIZATION FOR HEALTH CHATBOT STATE
# ---------------------------------------------------------
if "health_chat_input" not in st.session_state:
    st.session_state["health_chat_input"] = ""

if "health_chat_history" not in st.session_state:
    st.session_state["health_chat_history"] = []


# =======================================================
# DATA LOADING & PREPROCESSING
# =======================================================
@st.cache_data
def load_data():
    """
    Load and preprocess the Pima Indians Diabetes dataset.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        "preg", "glucose", "bp", "skin", "insulin",
        "bmi", "pedigree", "age", "class"
    ]
    df = pd.read_csv(url, names=cols)

    # Replace impossible zero values with NaN for selected columns
    zero_cols = ["glucose", "bp", "skin", "insulin", "bmi"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # Fill missing values with median of each column
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

    # Feature engineering
    df["bmi_age"] = df["bmi"] * df["age"]
    df["glucose_bmi"] = df["glucose"] * df["bmi"]
    df["high_bp"] = (df["bp"] >= 130).astype(int)

    return df


df = load_data()

# =======================================================
# FEATURES, SCALING & TRAIN/TEST SPLIT
# =======================================================
FEATURE_COLS = [
    "preg", "glucose", "bp", "skin", "insulin",
    "bmi", "pedigree", "age", "bmi_age", "glucose_bmi", "high_bp"
]
TARGET_COL = "class"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =======================================================
# MODEL TRAINING FUNCTION (CACHED)
# =======================================================
@st.cache_resource
def train_models():
    """Train multiple models and return them with their metrics."""
    models = {}
    metrics = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
    log_reg.fit(X_train, y_train)
    models["Logistic Regression"] = log_reg

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # XGBoost
    xgb = XGBClassifier(eval_metric="logloss")
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    # Evaluate all models
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Probabilities (if available)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        metrics[name] = (acc, auc)

    return models, metrics


models, model_metrics = train_models()

# =======================================================
# SIDEBAR: LOGO + TEAM
# =======================================================
st.sidebar.image("futurizestudio_logo.jpeg", width=140)

st.sidebar.title("Project Team")
st.sidebar.markdown(
    "**Team Name:** Futurize Academy | Zenith-Trident Team"
)

st.sidebar.markdown(
    "**Developers:**\n"
    "- Joseph Duruh — Lead Developer (AI / ML)\n"
    "- Nasisira Seezibella — IT Infrastructure & Systems\n"
    "- Chimyzerem Janet Uche-Ukah — Software Developer (Cloud, Frontend)"
)

# =======================================================
# SIDEBAR: HEALTH ASSISTANT (BETA)
# =======================================================
with st.sidebar.expander("Health Assistant (beta)", expanded=True):

    st.write(
        "Ask simple questions about diabetes risk factors, lifestyle, "
        "or what the numbers on this page mean."
    )
    st.info("⚠️ This assistant cannot give medical advice or diagnoses.")

    # User input box (bound to session_state)
    user_question = st.text_area(
        "Your question",
        key="health_chat_input",
        placeholder="E.g. What does 'preg' mean? How is BMI related to risk?"
    )

    # Send Button
    if st.button("Send", key="send_button"):
        question = user_question.strip()

        if question:
            # Generate answer with our helper
            answer = generate_health_answer(question)

            # Append to chat history
            st.session_state.health_chat_history.append(
                {"q": question, "a": answer}
            )

            # Clear the text box
            st.session_state["health_chat_input"] = ""

    # Display conversation history
    if st.session_state.health_chat_history:
        st.markdown("### Assistant Response")
        for chat in st.session_state.health_chat_history:
            st.write(f"**You:** {chat['q']}")
            st.write(f"**Assistant:** {chat['a']}")
            st.write("---")

# =======================================================
# SIDEBAR: PATIENT INPUTS + RESET
# =======================================================

# Default values for inputs
default_inputs = {
    "preg": 2,
    "glucose": 120,
    "bp": 70,
    "skin": 25,
    "insulin": 80,
    "bmi": 28.0,
    "pedigree": 0.5,
    "age": 35,
}

# Initialize patient inputs once
if "inputs_initialized" not in st.session_state:
    for k, v in default_inputs.items():
        st.session_state[k] = v
    st.session_state["inputs_initialized"] = True

st.sidebar.header("Patient Data")

# Reset button
if st.sidebar.button("Reset inputs"):
    for k, v in default_inputs.items():
        st.session_state[k] = v

# Capture inputs
preg = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=int(st.session_state["preg"]),
    step=1,
    key="preg"
)

glucose = st.sidebar.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=int(st.session_state["glucose"]),
    step=1,
    key="glucose"
)

bp = st.sidebar.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=int(st.session_state["bp"]),
    step=1,
    key="bp"
)

skin = st.sidebar.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=int(st.session_state["skin"]),
    step=1,
    key="skin"
)

insulin = st.sidebar.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=int(st.session_state["insulin"]),
    step=1,
    key="insulin"
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=float(st.session_state["bmi"]),
    step=0.1,
    key="bmi"
)

pedigree = st.sidebar.number_input(
    "Diabetes Pedigree",
    min_value=0.0,
    max_value=2.5,
    value=float(st.session_state["pedigree"]),
    step=0.01,
    key="pedigree"
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=90,
    value=int(st.session_state["age"]),
    step=1,
    key="age"
)

# Derived features based on inputs
bmi_age = bmi * age
glucose_bmi = glucose * bmi
high_bp = 1 if bp >= 130 else 0

# =======================================================
# MAIN LAYOUT: TITLE + PERFORMANCE + ABOUT
# =======================================================

# Top logo in main area
st.image("futurizestudio_logo.jpeg", width=150)

st.title("African Diabetes Risk Predictor")
st.markdown(
    "A prototype clinical decision support tool using multiple "
    "machine learning algorithms on the Pima Indians Diabetes dataset."
)

perf_col, about_col = st.columns([2, 1])

with perf_col:
    st.subheader("Model Performance")

    model_name = st.selectbox(
        "Choose model",
        options=list(models.keys()),
        index=0
    )

    sel_acc, sel_auc = model_metrics[model_name]

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{sel_acc:.2%}")
    c2.metric("ROC-AUC", f"{sel_auc:.3f}")

with about_col:
    with st.expander("About the model and data", expanded=False):
        st.markdown(
            "- Dataset: **Pima Indians Diabetes** (768 samples, 8 clinical features)\n"
            "- Target: Binary outcome indicating diabetes\n"
            "- Core features: pregnancies, glucose, blood pressure, skin thickness, "
            "insulin, BMI, diabetes pedigree, and age.\n"
            "- Engineered features: BMI × Age, Glucose × BMI, and a high blood "
            "pressure flag.\n\n"
            "This app is for **research and education**. It does **not** replace "
            "medical diagnosis or professional judgement."
        )

# =======================================================
# PATIENT INPUT SUMMARY
# =======================================================
st.subheader("Patient Input Summary")

patient_row = pd.DataFrame(
    [{
        "preg": preg,
        "glucose": glucose,
        "bp": bp,
        "skin": skin,
        "insulin": insulin,
        "bmi": bmi,
        "pedigree": pedigree,
        "age": age,
        "bmi_age": bmi_age,
        "glucose_bmi": glucose_bmi,
        "high_bp": high_bp,
    }]
)

st.dataframe(
    patient_row,
    use_container_width=True
)

# =======================================================
# PREDICTION
# =======================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

predict_button = st.button("Predict")

if predict_button:
    # Scale patient input
    X_new = scaler.transform(patient_row[FEATURE_COLS])

    model = models[model_name]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_new)[0, 1])
    else:
        # Convert decision function output to a probability-like value
        score = model.decision_function(X_new)[0]
        prob = 1 / (1 + np.exp(-score))

    risk_score = prob * 100

    # Risk categories
    if prob < 0.33:
        risk_level = "Low"
    elif prob < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Store in history
    record = patient_row.copy()
    record["model"] = model_name
    record["risk_probability"] = prob
    record["risk_level"] = risk_level
    st.session_state["history"].append(record.iloc[0].to_dict())

    # Results card
    st.subheader("Prediction Result")

    # Progress bar (0–100%)
    st.write("Estimated risk probability")
    st.progress(min(max(prob, 0.0), 1.0))

    # Color-coded message
    msg = f"Risk probability: {risk_score:.1f}% — {risk_level} risk"

    if risk_level == "Low":
        st.success(msg)
    elif risk_level == "Medium":
        st.warning(msg)
    else:
        st.error(msg)

    # Summary metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk level", risk_level)
    m2.metric("Risk score (0–100)", f"{risk_score:.1f}")
    m3.metric("Model used", model_name)

# =======================================================
# HISTORY + DOWNLOAD
# =======================================================
st.subheader("Prediction History and Export")

if st.session_state["history"]:
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df, use_container_width=True, height=250)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv,
        file_name="diabetes_predictions_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet. Enter patient data and click **Predict** to see results.")

# =======================================================
# FOOTER
# =======================================================
st.markdown("---")
st.markdown(
    "This tool is an **experimental prototype** for educational and research purposes only. "
    "It is **not** intended for clinical use or to guide real-world medical decisions."
)
