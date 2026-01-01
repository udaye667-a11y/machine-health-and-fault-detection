import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ PAGE SETUP ------------------ #
st.set_page_config(page_title="Smart Machine Health & Fault Detection", layout="centered")
st.title("🛠 Smart Machine Health & Fault Detection System")

# Load model
model, feature_columns = joblib.load("model.pkl")

# ------------------ SAMPLE DOWNLOAD SECTION ------------------ #
st.subheader("📁 Download Sample Data")
colS1, colS2 = st.columns(2)

if os.path.exists("feature_time_48k_2048_load_1.csv"):
    with open("feature_time_48k_2048_load_1.csv", "rb") as f:
        colS1.download_button(
            label="⬇️ Sample Feature CSV",
            data=f,
            file_name="sample_features.csv",
            mime="text/csv"
        )

if os.path.exists("CWRU_sample_signal.npz"):
    with open("CWRU_sample_signal.npz", "rb") as f:
        colS2.download_button(
            label="⬇️ Sample Vibration (.npz)",
            data=f,
            file_name="sample_vibration.npz"
        )

uploaded_feat = st.file_uploader("📌 Upload Your Feature CSV", type=["csv"])
uploaded_npz = st.file_uploader("📌 Upload Your Raw Vibration (.npz)", type=["npz"])


# ------------------ FAULT PREDICTION ------------------ #
if uploaded_feat is not None:
    st.subheader("🔍 Fault Classification from Extracted Features")

    df = pd.read_csv(uploaded_feat)
    df = df[feature_columns]

    preds = model.predict(df)
    result_df = pd.DataFrame(preds, columns=["Predicted Condition"])
    st.dataframe(result_df)

    # Confidence
    st.subheader("📊 Prediction Confidence (%)")
    probs = model.predict_proba(df)
    conf_scores = np.max(probs, axis=1) * 100

    df_conf = pd.DataFrame({
        "Predicted Condition": preds,
        "Confidence (%)": conf_scores.round(2)
    })
    st.dataframe(df_conf)

    # Status Message
    st.subheader("🧠 Machine Health Status")
    unique_faults = result_df["Predicted Condition"].unique()
    if "Normal" in unique_faults and len(unique_faults) == 1:
        st.success("🟢 Machine is Healthy")
    elif "Normal" in unique_faults and len(unique_faults) > 1:
        st.warning("🟡 Mixed Conditions — Maintenance Suggested!")
    else:
        st.error("🔴 Fault Detected — Immediate Inspection Required!")

    # ------------------ Dashboard ------------------ #
    st.subheader("📈 Machine Health Dashboard")
    total = len(df_conf)
    normal_count = sum(df_conf["Predicted Condition"] == "Normal")
    health_score = (normal_count / total) * 100

    st.metric("Overall Machine Health", f"{health_score:.2f}%")

    fault_counts = df_conf["Predicted Condition"].value_counts()
    fig_dash, ax_dash = plt.subplots()
    ax_dash.pie(fault_counts, labels=fault_counts.index, autopct="%1.1f%%", startangle=90)
    ax_dash.set_title("Fault Distribution")
    st.pyplot(fig_dash)


# ------------------ RAW SIGNAL VISUALIZATION + FAULT MARKERS ------------------ #
if uploaded_npz is not None:
    st.subheader("📈 Time Domain Vibration Signal")

    data_npz = np.load(uploaded_npz)
    signal = data_npz["signal"]
    if signal.ndim > 1: signal = signal.flatten()

    fig1, ax1 = plt.subplots()
    ax1.plot(signal[:3000])
    st.pyplot(fig1)

    st.subheader("📊 FFT Spectrum + Fault Indicators")
    sampling_rate = 48000
    fft = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    mask = freq >= 0
    freq, fft = freq[mask], fft[mask]

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:5000], fft[:5000])

    # CWRU Bearing fault frequencies
    shaft_freq = 30
    markers = {"BPFI": 5.4*shaft_freq, "BPFO": 3.6*shaft_freq, "FTF": 0.4*shaft_freq}
    for label, val in markers.items():
        ax2.axvline(val, color="red", linestyle="--")
        ax2.text(val, max(fft[:5000]) * 0.6, label, rotation=90, color="red")

    st.pyplot(fig2)


# ------------------ CONFUSION MATRIX ------------------ #
st.subheader("📌 Model Confusion Matrix")
if "cm.npy" in os.listdir() and "labels.npy" in os.listdir():
    cm = np.load("cm.npy")
    labels = np.load("labels.npy", allow_pickle=True)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Greens", xticklabels=labels, yticklabels=labels)
    st.pyplot(fig3)
else:
    st.warning("Confusion matrix unavailable ❗")


st.write("---")
st.caption("© Smart Machine Health System | Streamlit Deployment 🚀")
