import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the trained ML model and feature names
model, feature_columns = joblib.load("model.pkl")

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Smart Machine Monitoring", layout="centered")
st.title("🛠 Smart Machine Condition Monitoring System")

st.markdown("Upload real-time machine data below to detect faults and visualize vibration analysis.")

uploaded_feat = st.file_uploader("📌 Upload Feature CSV", type=["csv"])
uploaded_npz = st.file_uploader("📌 Upload Raw Vibration (.npz)", type=["npz"])

# ------------------ FAULT PREDICTION ------------------ #
if uploaded_feat is not None:
    st.subheader("🔍 Fault Classification from Extracted Features")
    df = pd.read_csv(uploaded_feat)
    df = df[feature_columns]

    preds = model.predict(df)
    st.dataframe(pd.DataFrame(preds, columns=["Predicted Condition"]))

    # Confidence Table
    st.subheader("📊 Prediction Confidence (%)")
    probs = model.predict_proba(df)
    conf_scores = np.max(probs, axis=1) * 100
    df_conf = pd.DataFrame({
        "Predicted Condition": preds,
        "Confidence (%)": conf_scores.round(2)
    })
    st.dataframe(df_conf)

    # Machine Health Indicator
    st.subheader("🧠 Machine Health Status")
    unique_faults = df_conf["Predicted Condition"].unique()

    if "Normal" in unique_faults and len(unique_faults) == 1:
        st.success("🟢 Machine is Healthy")
    elif "Normal" in unique_faults and len(unique_faults) > 1:
        st.warning("🟡 Mixed Conditions Detected — Recommend Maintenance Check!")
    else:
        st.error("🔴 Fault Detected — Immediate Inspection Required!")

    st.success("Prediction Completed Successfully! 🚀")

# ------------------ VIBRATION SIGNAL PLOTS ------------------ #
if uploaded_npz is not None:
    st.subheader("📈 Time Domain & Frequency Analysis")
    npz_data = np.load(uploaded_npz)
    signal = npz_data["signal"]

    st.write("Signal shape:", signal.shape)

    # Auto-fix multi-dimensional inputs
    if signal.ndim == 2:
        signal = signal[:, 0]
    elif signal.ndim == 3:
        signal = signal[:, 0, 0]

    st.write("Adjusted signal shape:", signal.shape)

    # Plot Time-Domain Signal
    fig1, ax1 = plt.subplots()
    ax1.plot(signal[:3000])
    ax1.set_title("Time Domain – Vibration Waveform")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # FFT Plot
    fft = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), d=1/48000)
    mask = freq >= 0
    freq = freq[mask]
    fft = fft[mask]

    st.write("FFT length:", len(fft))

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[:5000], fft[:5000])
    ax2.set_title("Frequency Domain – FFT Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.success("Graphs Generated Successfully 🎯")

# ------------------ CONFUSION MATRIX ------------------ #
st.subheader("📌 Model Confusion Matrix")

df_cm = pd.read_csv(r"C:\Users\Admin\Downloads\cwru-dataset\feature_time_48k_2048_load_1.csv")
df_cm['fault'] = df_cm['fault'].apply(
    lambda x: 'Ball Fault' if 'Ball' in x else
              'Inner Race Fault' if 'IR' in x else
              'Outer Race Fault' if 'OR' in x else
              'Normal'
)
X_cm = df_cm[feature_columns]
y_cm = df_cm['fault']
y_pred_cm = model.predict(X_cm)
cm = confusion_matrix(y_cm, y_pred_cm)

fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_title("Confusion Matrix")
st.pyplot(fig3)
