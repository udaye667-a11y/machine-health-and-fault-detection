import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="AI-Driven Predictive Maintenance System",
    layout="wide",
    page_icon="🛠"
)

# Dark Theme Branding Header
st.markdown("""
<style>
.header-container {
    background-color: #0A0A0A;
    padding: 20px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.title-text {
    color: #00C3FF;
    font-size: 26px;
    font-weight: bold;
}
.subtitle-text {
    color: #BBBBBB;
    font-size: 16px;
}
.details-text {
    color: #FFFFFF;
    font-size: 14px;
    line-height: 1.3;
}
</style>
""", unsafe_allow_html=True)

# College Logo (GitHub raw link version - update after uploading logo file)
logo_url = "https://raw.githubusercontent.com/udaye667-a11y/machine-health-and-fault-detection/main/dsatm_logo.png"

st.markdown(f"""
<div class="header-container">
    <div>
        <div class="title-text">AI-Driven Predictive Maintenance & Condition Monitoring System</div>
        <div class="subtitle-text">For Rotating Machinery Using Machine Learning & Vibration Analysis</div><br>
        <div class="details-text">
            <b>Student:</b> Uday Eshwar<br>
            <b>USN:</b> 1DT23ME017<br>
            <b>Department:</b> Mechanical Engineering<br>
            <b>Institution:</b> Dayananda Sagar Academy of Technology & Management, Bengaluru<br>
            <b>Academic Year:</b> 2024 – 2025
        </div>
    </div>
    <div>
        <img src="{logo_url}" width="120">
    </div>
</div>
""", unsafe_allow_html=True)


st.write("")


# Load model
model, feature_columns = joblib.load("model.pkl")

# ------------------ Sample Download ------------------ #
st.subheader("📥 Download Sample Data")
col1, col2 = st.columns(2)

if os.path.exists("feature_time_48k_2048_load_1.csv"):
    with open("feature_time_48k_2048_load_1.csv", "rb") as f:
        col1.download_button("⬇️ Sample Feature CSV", f, "sample_features.csv")

if os.path.exists("CWRU_sample_signal.npz"):
    with open("CWRU_sample_signal.npz", "rb") as f:
        col2.download_button("⬇️ Sample Vibration Signal", f, "sample_signal.npz")


# File Upload
uploaded_feat = st.file_uploader("📌 Upload Your Feature CSV", type=["csv"])
uploaded_npz = st.file_uploader("📌 Upload Your Raw Vibration (.npz)", type=["npz"])


# ------------------ Prediction Section ------------------ #
if uploaded_feat is not None:
    st.subheader("🧠 Fault Classification Results")

    df = pd.read_csv(uploaded_feat)
    df = df[feature_columns]

    preds = model.predict(df)
    df_pred = pd.DataFrame(preds, columns=["Predicted Condition"])
    st.dataframe(df_pred)

    # Confidence
    probs = model.predict_proba(df)
    conf_scores = np.max(probs, axis=1) * 100

    df_conf = pd.DataFrame({
        "Predicted Condition": preds,
        "Confidence (%)": conf_scores.round(2)
    })
    st.write(df_conf)

    # Health Score
    total = len(df_conf)
    normal_count = sum(df_conf["Predicted Condition"] == "Normal")
    health_score = (normal_count / total) * 100

    st.metric("Machine Health Score", f"{health_score:.2f}%")

    st.success("Prediction Completed Successfully! 🚀")


# ------------------ Signal Visualization ------------------ #
if uploaded_npz is not None:
    st.subheader("📈 Time Domain Vibration Signal")

    npz = np.load(uploaded_npz)
    signal = npz["signal"]
    if signal.ndim > 1: signal = signal.flatten()

    fig1, ax1 = plt.subplots()
    ax1.plot(signal[:3000])
    st.pyplot(fig1)

    # FFT
    st.subheader("📊 Frequency Spectrum with Fault Indicators")
    fft = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), d=1/48000)
    m = freq >= 0
    freq, fft = freq[m], fft[m]

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:5000], fft[:5000])

    # Fault markers
    shaft = 30
    markers = {"BPFI": 5.4 * shaft, "BPFO": 3.6 * shaft, "FTF": 0.4 * shaft}
    for name, val in markers.items():
        ax2.axvline(val, color="red", linestyle="--")
        ax2.text(val, max(fft[:5000])*0.7, name, rotation=90, color="red")

    st.pyplot(fig2)

    st.success("Signal Analysis Completed 🎯")


# ------------------ Confusion Matrix ------------------ #
st.subheader("📌 Confusion Matrix")

if "cm.npy" in os.listdir() and "labels.npy" in os.listdir():
    cm = np.load("cm.npy")
    labels = np.load("labels.npy", allow_pickle=True)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    st.pyplot(fig3)
else:
    st.warning("Confusion Matrix not found ❗")


st.write("---")
st.caption("© AI-Driven Predictive Maintenance System | Developed by: Uday Eshwar 🚀")
