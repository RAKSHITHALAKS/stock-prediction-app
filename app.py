import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
import logging

import os
import gdown
import tensorflow as tf
import joblib

# Folder to store models
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ------------------------------
# LSTM Model
# ------------------------------
lstm_url = "https://drive.google.com/uc?export=download&id=1JvhPw4mPvL7UWrGm1mwet3UBT5INjNXO"
lstm_path = os.path.join(model_dir, "lstm_stock.keras")
if not os.path.exists(lstm_path):
    gdown.download(lstm_url, lstm_path, quiet=False)
lstm_model = tf.keras.models.load_model(lstm_path)

# ------------------------------
# MLP Model
# ------------------------------
mlp_url = "https://drive.google.com/uc?export=download&id=12FtUiL_PKXfo1Z6Nv7adds3NOta_NICr"
mlp_path = os.path.join(model_dir, "mlp_model.pkl")
if not os.path.exists(mlp_path):
    gdown.download(mlp_url, mlp_path, quiet=False)
mlp_model = joblib.load(mlp_path)

# ------------------------------
# SVM Model
# ------------------------------
svm_url = "https://drive.google.com/uc?export=download&id=1bOhNKntdNX5xEv5kv33QQKrbdiDSaiI7"
svm_path = os.path.join(model_dir, "svm_model.pkl")
if not os.path.exists(svm_path):
    gdown.download(svm_url, svm_path, quiet=False)
svm_model = joblib.load(svm_path)


# ------------------------
st.title("📊 Stock Price Prediction Dashboard")
st.markdown("""
Upload a CSV file with a **`close`** column to visualize stock trends and predictions.
Select a model and explore interactive analytics and portfolio tracking.
""")

# ------------------------
# Upload CSV
# ------------------------
uploaded_file = st.file_uploader("Upload CSV (must contain 'close')", type="csv")
model_choice = st.radio("Choose a model:", ["SVM", "MLP", "LSTM"])

# ------------------------
# Helper Functions
# ------------------------
def process_lstm(df):
    seq_len = 60
    input_shape = lstm_model.input_shape
    n_features = input_shape[-1]

    if n_features == 1:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[["close"]])
    else:
        required = ["open", "high", "low", "close", "volume"]
        if not all(f in df.columns for f in required):
            st.error(f"LSTM requires {required}.")
            st.stop()
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[required])

    X_seq = []
    for i in range(len(data_scaled) - seq_len):
        X_seq.append(data_scaled[i:i+seq_len])
    X_seq = np.array(X_seq)

    pred_scaled = lstm_model.predict(X_seq)
    if n_features > 1:
        pred = scaler.inverse_transform(np.hstack([pred_scaled, np.zeros((len(pred_scaled), n_features-1))]))[:, 0]
    else:
        pred = scaler.inverse_transform(pred_scaled)

    y_true = df["close"].values[seq_len:]
    return y_true, pred

def process_svm(df):
    window_size = 5
    y_true = df["close"].values
    X = []
    for i in range(len(y_true) - window_size):
        X.append(y_true[i:i+window_size])
    X = np.array(X)
    y_aligned = y_true[window_size:]

    # Dynamic scaling
    X_scaled = []
    for window in X:
        scaler = StandardScaler()
        X_scaled.append(scaler.fit_transform(window.reshape(-1,1)).flatten())
    X_scaled = np.array(X_scaled)

    pred = svm_model.predict(X_scaled)
    pred_direction = ["Down" if p==0 else "Up" for p in pred]

    # Metrics
    y_true_dir = y_aligned[1:] > y_aligned[:-1]
    y_true_dir = y_true_dir.astype(int)
    y_pred_dir = pred[:len(y_true_dir)]

    return y_aligned, pred_direction, accuracy_score(y_true_dir, y_pred_dir), confusion_matrix(y_true_dir, y_pred_dir), classification_report(y_true_dir, y_pred_dir)

def process_mlp(df):
    window_size = 5
    y_true = df["close"].values
    X = []
    for i in range(len(y_true) - window_size):
        X.append(y_true[i:i+window_size])
    X = np.array(X)
    y_aligned = y_true[window_size:]

    pred = mlp_model.predict(X)
    return y_aligned, pred

# ------------------------
# Main App Logic
# ------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    logging.info(f"User uploaded file: {uploaded_file.name}, selected model: {model_choice}")

    if "close" not in df.columns:
        st.error("CSV must have a 'close' column.")
    else:
        st.subheader("📌 Data Preview")
        st.dataframe(df.head())

        if model_choice == "LSTM":
            y_true, pred = process_lstm(df)
            st.subheader(" Predicted Prices (LSTM)")
            st.line_chart(pred)

            st.subheader(" Predicted vs Actual")
            combined = pd.DataFrame({"Actual": y_true.flatten(), "Predicted": pred.flatten()})
            st.line_chart(combined)

            st.subheader(" Model Performance")
            st.write(f"**MSE:** {mean_squared_error(y_true, pred):.4f}")
            st.write(f"**MAE:** {mean_absolute_error(y_true, pred):.4f}")

        elif model_choice == "SVM":
            y_true, pred_direction, acc, cm, cr = process_svm(df)
            st.subheader(" Predicted Direction (SVM)")
            st.write(pred_direction)
            st.subheader(" Predicted Direction Chart")
            st.line_chart([0 if p=="Down" else 1 for p in pred_direction])
            st.subheader(" SVM Classification Metrics")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write("**Confusion Matrix:**")
            st.write(cm)
            st.write("**Classification Report:**")
            st.text(cr)

        elif model_choice == "MLP":
            y_true, pred = process_mlp(df)
            st.subheader(" Predicted Prices (MLP)")
            st.line_chart(pred)
            st.subheader(" Predicted vs Actual")
            combined = pd.DataFrame({"Actual": y_true.flatten(), "Predicted": pred.flatten()})
            st.line_chart(combined)
            st.subheader(" Model Performance")
            st.write(f"**MSE:** {mean_squared_error(y_true, pred):.4f}")
            st.write(f"**MAE:** {mean_absolute_error(y_true, pred):.4f}")

        # ------------------------
        # Downloadable Report
        # ------------------------
        if model_choice in ["LSTM", "MLP"]:
            csv = combined.to_csv(index=False)
            st.download_button(
                label=" Download Predictions as CSV",
                data=csv,
                file_name=f"{model_choice}_predictions.csv",
                mime="text/csv"
            )
