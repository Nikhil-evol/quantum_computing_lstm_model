
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import pennylane as qml
from pennylane import numpy as pnp

MODEL_DIR = "saved_models"
MODEL_NAME = "hybrid_quantum_lstm_stock_model.keras"
WEIGHTS_NAME = "quantum_weights.npy"

# ------------------- Data -------------------
def download_stock_data(ticker="SBUX", period="7y"):
    df = yf.download(ticker, period=period, interval="1d")[['Close']]
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_dates)
    df.ffill(inplace=True)
    df.index.name = 'Date'
    return df

def create_dataset(series, window_size, horizon=7):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:(i+window_size)])
        y.append(series[(i+window_size):(i+window_size+horizon)].reshape(-1))
    return np.array(X), np.array(y)

# ------------------- Quantum -------------------
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
    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes

def reduce_window_to_qubits(window, n_qubits):
    window = window.flatten()
    idx = np.linspace(0, len(window)-1, n_qubits).astype(int)
    return window[idx]

def compute_quantum_features_for_dataset(X_windows, circuit, weights, n_qubits):
    features = []
    for i in range(X_windows.shape[0]):
        window = X_windows[i]
        reduced = reduce_window_to_qubits(window, n_qubits)
        angles = reduced * np.pi
        q_out = circuit(pnp.array(angles), weights)
        features.append(np.array(q_out, dtype=np.float64))
    return np.array(features)

# ------------------- LSTM -------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(7)  # predict next 7 days
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model

# ------------------- Training -------------------
def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info logs

    df = download_stock_data("SBUX", "5y")
    # df.to_csv('SBUX.csv')
    df.to_csv("SBUX.csv", index=True, index_label="Date")

    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(df.values)

    train_size = int(len(scaled_data)*0.7)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    window_size, horizon = 365, 7

    X_train_raw, y_train = create_dataset(train_data, window_size, horizon)
    X_test_raw, y_test = create_dataset(test_data, window_size, horizon)

    n_qubits, n_layers = 4, 2
    circuit, weight_shapes = build_quantum_feature_map(n_qubits, n_layers)
    q_weights = pnp.random.uniform(0, np.pi, size=weight_shapes['weights'], requires_grad=False)

    print("Computing quantum features...")
    X_train_q = compute_quantum_features_for_dataset(X_train_raw, circuit, q_weights, n_qubits)
    X_test_q = compute_quantum_features_for_dataset(X_test_raw, circuit, q_weights, n_qubits)

    X_train = X_train_q.reshape((X_train_q.shape[0], 1, n_qubits))
    X_test = X_test_q.reshape((X_test_q.shape[0], 1, n_qubits))

    model = build_lstm_model((1, n_qubits))
    tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[tb_callback])

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    np.save(os.path.join(MODEL_DIR, WEIGHTS_NAME), np.array(q_weights))
    print("✅ Model, scaler, and quantum weights saved successfully.")

if __name__=="__main__":
    main()
