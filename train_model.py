import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("feature_time_48k_2048_load_1.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# -------------------------------------------------
feature_columns = [
    "max", "min", "mean", "sd", "rms",
    "skewness", "kurtosis", "crest", "form"
]

X = df[feature_columns]
y = df["fault"]

# -------------------------------------------------
# 3. FEATURE SCALING
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 5. TRAIN RANDOM FOREST MODEL
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 6. MODEL EVALUATION
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -------------------------------------------------
# 7. SAVE MODEL AND REQUIRED FILES
# -------------------------------------------------

# Save model, features, and scaler together
joblib.dump(
    {
        "model": model,
        "features": feature_columns,
        "scaler": scaler
    },
    "model.pkl"
)

# Save confusion matrix and labels
np.save("confusion_matrix.npy", cm)
np.save("labels.npy", model.classes_)

# Save feature importance
np.save("feature_importance.npy", model.feature_importances_)

# Save classification report
with open("classification_report.txt", "w") as f:
    f.write(report)

print("\n✔ Training Complete")
print("✔ model.pkl saved")
print("✔ confusion_matrix.npy saved")
print("✔ labels.npy saved")
print("✔ feature_importance.npy saved")
print("✔ classification_report.txt saved")
