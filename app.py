import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Load Models ---
lstm_model = tf.keras.models.load_model("lstm_stock.keras")
svm_model = joblib.load("svm_model.pkl")
mlp_model = joblib.load("mlp_model.pkl")

st.title(" Stock Price Prediction (SVM, MLP, LSTM)")

st.markdown("""
Upload a CSV file with a **`close`** column to predict stock prices.
You can select which model to use and compare the predictions.
""")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV (must contain 'close')", type="csv")

# --- Model Selection ---
model_choice = st.radio("Choose a model:", ["SVM", "MLP", "LSTM"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "close" not in df.columns:
        st.error("CSV must have a 'close' column.")
    else:
        st.subheader(" Data Preview")
        st.dataframe(df.head())

        y_true = df["close"].values

        if model_choice == "LSTM":
            # --- Prepare data for LSTM ---
            seq_len = 60

            # Use a new MinMaxScaler fitted on the uploaded data
            scaler_lstm = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler_lstm.fit_transform(df[["close"]])

            X_seq = []
            for i in range(len(data_scaled) - seq_len):
                X_seq.append(data_scaled[i:i+seq_len])
            X_seq = np.array(X_seq)

            # Predictions
            pred_scaled = lstm_model.predict(X_seq)
            pred = scaler_lstm.inverse_transform(pred_scaled)

            # Align actual values
            y_true = y_true[seq_len:]

        else:
            # --- Prepare features for SVM / MLP using sliding window of 5 ---
            window_size = 5
            X = []
            for i in range(len(y_true) - window_size):
                X.append(y_true[i:i+window_size])
            X = np.array(X)

            # Align y_true for comparison
            y_aligned = y_true[window_size:]

            if model_choice == "SVM":
                # --- Dynamic scaling for each window ---
                from sklearn.preprocessing import StandardScaler
                X_scaled = []
                for window in X:
                    scaler = StandardScaler()
                    X_scaled.append(scaler.fit_transform(window.reshape(-1,1)).flatten())
                X_scaled = np.array(X_scaled)

                # --- Predict direction ---
                pred = svm_model.predict(X_scaled)

                # Convert 0/1 to Down/Up
                pred_direction = ["Down" if p == 0 else "Up" for p in pred]
                st.subheader(" Predicted Direction (SVM)")
                st.write(pred_direction)

                # Optional: step chart
                st.subheader(" Predicted Direction Chart")
                st.line_chart([0 if p=="Down" else 1 for p in pred_direction])

                # --- Compute classification metrics ---
                from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
                # Generate actual directions from y_aligned
                y_true_dir = y_aligned[1:] > y_aligned[:-1]
                y_true_dir = y_true_dir.astype(int)
                y_pred_dir = pred[:len(y_true_dir)]
                st.subheader(" SVM Classification Metrics")
                st.write(f"**Accuracy:** {accuracy_score(y_true_dir, y_pred_dir):.4f}")
                st.write("**Confusion Matrix:**")
                st.write(confusion_matrix(y_true_dir, y_pred_dir))
                st.write("**Classification Report:**")
                st.text(classification_report(y_true_dir, y_pred_dir))

            elif model_choice == "MLP":
                pred = mlp_model.predict(X)

            y_true = y_aligned

        # --- Show Predictions for MLP / LSTM ---
        if model_choice != "SVM":
            st.subheader(f" Predicted Prices ({model_choice})")
            st.line_chart(pred)

            # --- Compare Actual vs Predicted ---
            st.subheader(" Predicted vs Actual")
            combined = pd.DataFrame({
                "Actual": y_true.flatten(),
                "Predicted": pred.flatten()
            })
            st.line_chart(combined)

            # --- Show Metrics ---
            mse = mean_squared_error(y_true, pred)
            mae = mean_absolute_error(y_true, pred)

            st.subheader(" Model Performance")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**MAE:** {mae:.4f}")
