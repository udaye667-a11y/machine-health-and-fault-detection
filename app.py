import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# ------------------ PAGE SETUP ------------------ #
st.set_page_config(page_title="Smart Machine Health & Fault Detection", layout="centered")
st.title("🛠 Smart Machine Health & Fault Detection System")
st.write("Developed using Machine Learning & Vibration Analysis")

# Load trained model & feature names
model, feature_columns = joblib.load("model.pkl")

uploaded_feat = st.file_uploader("📌 Upload Feature CSV", type=["csv"])
uploaded_npz = st.file_uploader("📌 Upload Raw Vibration (.npz)", type=["npz"])

# ------------------ FAULT PREDICTION ------------------ #
if uploaded_feat is not None:
    st.subheader("🔍 Fault Classification from Extracted Features")

    df = pd.read_csv(uploaded_feat)
    df = df[feature_columns]

    preds = model.predict(df)
    st.dataframe(pd.DataFrame(preds, columns=["Predicted Condition"]))

    # Confidence Score
    st.subheader("📊 Prediction Confidence (%)")
    probs = model.predict_proba(df)
    conf_scores = np.max(probs, axis=1) * 100
    df_conf = pd.DataFrame({
        "Predicted Condition": preds,
        "Confidence (%)": conf_scores.round(2)
    })
    st.dataframe(df_conf)

    # Health Indicator
    st.subheader("🧠 Machine Health Status")
    unique_faults = df_conf["Predicted Condition"].unique()

    if "Normal" in unique_faults and len(unique_faults) == 1:
        st.success("🟢 Machine is Healthy")
    elif "Normal" in unique_faults and len(unique_faults) > 1:
        st.warning("🟡 Mixed Conditions Detected — Recommend Maintenance Check!")
    else:
        st.error("🔴 Fault Detected — Immediate Inspection Required!")

    st.success("Prediction Completed Successfully! 🚀")


# ------------------ SIGNAL VISUALIZATION ------------------ #
if uploaded_npz is not None:
    st.subheader("📈 Raw Vibration Signal (Time Domain)")
    npz_data = np.load(uploaded_npz)
    signal = npz_data["signal"]

    st.write("Signal shape:", signal.shape)

    # Flatten if multidimensional
    if signal.ndim == 2:
        signal = signal[:, 0]
    elif signal.ndim == 3:
        signal = signal[:, 0, 0]

    fig1, ax1 = plt.subplots()
    ax1.plot(signal[:3000])
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Vibration Waveform")
    st.pyplot(fig1)

    # FFT
    st.subheader("📊 Frequency Spectrum (FFT)")
    fft = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), d=1/48000)
    mask = freq >= 0
    freq = freq[mask]
    fft = fft[mask]

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:5000], fft[:5000])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("FFT Spectrum")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.success("Graphs Generated Successfully 🎯")


# ------------------ CONFUSION MATRIX ------------------ #
st.subheader("📌 Model Confusion Matrix (Evaluation Results)")

if "cm.npy" in os.listdir() and "labels.npy" in os.listdir():
    cm = np.load("cm.npy")
    labels = np.load("labels.npy", allow_pickle=True)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels,
                yticklabels=labels)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    st.pyplot(fig3)
else:
    st.warning("Confusion matrix not available in cloud version ❗")


# ------------------ FOOTER ------------------ #
st.write("---")
st.caption("© Machine Health & Fault Detection System - Powered by Streamlit")
