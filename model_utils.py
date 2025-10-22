
import os
import joblib
import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp

MODEL_DIR = "saved_models"
MODEL_NAME = "hybrid_quantum_lstm_stock_model.keras"
WEIGHTS_NAME = "quantum_weights.npy"

# ---------------- Quantum ----------------
def build_quantum_feature_map(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l,i], wires=i)
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i,i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

def reduce_window_to_qubits(window, n_qubits):
    window = window.flatten()
    idx = np.linspace(0, len(window)-1, n_qubits).astype(int)
    return window[idx]

def quantum_preprocess(window, circuit, weights, scaler, n_qubits):
    scaled = scaler.transform(window.reshape(-1,1)).flatten()
    reduced = reduce_window_to_qubits(scaled, n_qubits)
    angles = reduced * np.pi
    return np.array(circuit(pnp.array(angles), weights), dtype=np.float64)

# ---------------- Load ----------------
def load_model_and_scaler():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    weights_path = os.path.join(MODEL_DIR, WEIGHTS_NAME)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    if os.path.exists(weights_path):
        q_weights = np.load(weights_path, allow_pickle=True)
        q_weights = pnp.array(q_weights)
    else:
        print("⚠️ Quantum weights not found. Using random initialization.")
        n_qubits, n_layers = 4, 2
        q_weights = pnp.random.uniform(0, np.pi, size=(n_layers, n_qubits))

    return model, scaler, q_weights

# ---------------- Predict ----------------
def predict_n_days(model, scaler, q_weights, recent_data, window_size, n_days, n_qubits=4, n_layers=2):
    if n_days <= 0: return []

    circuit = build_quantum_feature_map(n_qubits, n_layers)
    recent_list = list(recent_data)
    preds_scaled = []

    for _ in range(n_days):
        window = np.array(recent_list[-window_size:])
        q_features = quantum_preprocess(window, circuit, q_weights, scaler, n_qubits)
        X_input = q_features.reshape(1,1,n_qubits)
        next_scaled = model.predict(X_input, verbose=0).flatten()[0]
        preds_scaled.append(next_scaled)
        next_unscaled = scaler.inverse_transform(np.array([[next_scaled]]))[0,0]
        recent_list.append(next_unscaled)

    preds_scaled = np.array(preds_scaled).reshape(-1,1)
    return scaler.inverse_transform(preds_scaled).flatten().tolist()
