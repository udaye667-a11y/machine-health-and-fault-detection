import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance System",
    layout="wide"
)

# --------------------------------------------------
# HEADER WITH COLLEGE LOGO (STREAMLIT-NATIVE)
# --------------------------------------------------
logo_url = "https://raw.githubusercontent.com/udaye667-a11y/machine-health-and-fault-detection/main/dsatm_logo.jpg"

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    <div style="background-color:#0A0A0A;padding:25px;border-radius:14px;">
        <h1 style="color:#00C3FF;margin-bottom:10px;">
            AI-Driven Predictive Maintenance and Condition Monitoring System
        </h1>
        <p style="color:#E8E8E8;font-size:16px;margin:0;">
            <b>Uday Eshwar</b> | USN: 1DT23ME017<br>
            Department of Mechanical Engineering<br>
            Dayananda Sagar Academy of Technology & Management
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image(logo_url, width=130)

st.write("")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
bundle = joblib.load("model.pkl")
model = bundle["model"]
features = bundle["features"]
scaler = bundle["scaler"]

model_used = False

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.subheader("📂 Upload Data Files")
csv_file = st.file_uploader("Upload Feature CSV", type=["csv"])
npz_file = st.file_uploader("Upload Vibration Signal (.npz)", type=["npz"])

# --------------------------------------------------
# PREDICTION & ANALYSIS
# --------------------------------------------------
if csv_file is not None:
    model_used = True

    df = pd.read_csv(csv_file)
    X = scaler.transform(df[features])

    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = np.max(probs, axis=1) * 100

    results = pd.DataFrame({
        "Predicted Condition": preds,
        "Confidence (%)": confidence.round(2)
    })

    st.subheader("🔍 Fault Classification Results")
    st.dataframe(results, use_container_width=True)

    # ---------------- PIE CHART ----------------
    st.subheader("📊 Fault Distribution")
    counts = results["Predicted Condition"].value_counts()

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax_pie.axis("equal")
    st.pyplot(fig_pie)

    # ---------------- HEALTH & SEVERITY ----------------
    fault_percent = 100 - (counts.get("Normal", 0) / len(results) * 100)
    health_score = 100 - fault_percent

    if health_score >= 90:
        severity, color = "🟢 Healthy", "green"
    elif health_score >= 75:
        severity, color = "🟡 Low Severity", "gold"
    elif health_score >= 60:
        severity, color = "🟠 Moderate Severity", "orange"
    elif health_score >= 40:
        severity, color = "🔴 High Severity", "red"
    else:
        severity, color = "🚨 CRITICAL", "darkred"

    st.subheader("🚦 Machine Health Status")
    st.markdown(f"<h3 style='color:{color}'>{severity}</h3>", unsafe_allow_html=True)
    st.progress(int(health_score))

    st.download_button(
        "⬇ Download Prediction Results",
        results.to_csv(index=False),
        "prediction_results.csv"
    )

# --------------------------------------------------
# FFT ANALYSIS
# --------------------------------------------------
if npz_file is not None:
    st.subheader("📈 Vibration Signal Analysis")

    signal = np.load(npz_file)["signal"].flatten()

    fig, ax = plt.subplots()
    ax.plot(signal[:3000])
    ax.set_title("Time Domain Signal")
    st.pyplot(fig)

    st.subheader("📊 Frequency Spectrum (FFT) – 1500 RPM")

    fft = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), 1 / 48000)
    mask = freq >= 0

    fig2, ax2 = plt.subplots()
    ax2.plot(freq[mask][:5000], fft[mask][:5000])

    shaft_freq = 25  # 1500 RPM
    fault_freqs = {
        "BPFI": 5.4 * shaft_freq,
        "BPFO": 3.6 * shaft_freq,
        "FTF": 0.4 * shaft_freq
    }

    for label, f in fault_freqs.items():
        ax2.axvline(f, color="red", linestyle="--")
        ax2.text(f, max(fft[:5000]) * 0.7, label, rotation=90, color="red")

    st.pyplot(fig2)

# --------------------------------------------------
# MODEL EVALUATION (ONLY AFTER UPLOAD)
# --------------------------------------------------
if model_used:

    if os.path.exists("feature_importance.npy"):
        st.subheader("🧠 Feature Importance")
        imp = np.load("feature_importance.npy")

        fig3, ax3 = plt.subplots()
        sns.barplot(x=imp, y=features, ax=ax3)
        st.pyplot(fig3)

    if os.path.exists("confusion_matrix.npy"):
        st.subheader("📌 Confusion Matrix")
        cm = np.load("confusion_matrix.npy")
        labels = np.load("labels.npy", allow_pickle=True)

        fig4, ax4 = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues"
        )
        st.pyplot(fig4)

    if os.path.exists("classification_report.txt"):
        st.subheader("📄 Classification Report")
        st.text(open("classification_report.txt").read())

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© 2024–25 | AI-Driven Predictive Maintenance System | Mechanical Engineering | DSATM")
