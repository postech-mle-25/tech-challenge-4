# src/train.py
import os

# ↓ silencia CUDA/XLA e força CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.data_loader import StockDataLoader


def create_sequences(data: np.ndarray, seq_len: int, target_idx: int = 3):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def build_lstm(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def main(args):
    print(f"🚀 Iniciando treinamento para {args.symbol}")
    print("📊 Carregando dados...")

    loader = StockDataLoader(
        args.symbol,
        args.start_date,
        args.end_date,
        allow_synthetic=not args.no_synthetic,
    )
    df = loader.fetch_data()
    df = loader.add_technical_indicators(df)
    print(f"✅ {len(df)} registros após indicadores (fonte={loader.get_source()})")

    print("🔧 Preprocessando dados...")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    features = df[cols].values.astype("float32")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    seq_len = args.sequence_length
    X, y = create_sequences(scaled, seq_len, target_idx=3)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"✅ Train: {X_train.shape}, Test: {X_val.shape}")
    print("🤖 Criando modelo LSTM...")
    model = build_lstm((seq_len, X.shape[2]))

    print("🏋️ Treinando modelo...")
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cb,
    )

    print("📈 Avaliando modelo...")
    preds_scaled = model.predict(X_val, verbose=0).reshape(-1, 1)

    n_features = getattr(scaler, "n_features_in_", 5)
    tmp = np.zeros((len(preds_scaled), n_features))
    tmp[:, 3] = preds_scaled.reshape(-1)

    y_tmp = np.zeros((len(y_val), n_features))
    y_tmp[:, 3] = y_val.reshape(-1)

    preds = scaler.inverse_transform(tmp)[:, 3]
    y_true = scaler.inverse_transform(y_tmp)[:, 3]

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = float(np.mean(np.abs((y_true - preds) / np.clip(np.abs(y_true), 1e-8, None))) * 100)

    print(f"✅ Métricas: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")

    print("💾 Salvando modelo e scaler...")
    os.makedirs("models/saved", exist_ok=True)
    try:
        model.save("models/saved/lstm_model.keras")
    except Exception:
        model.save("models/saved/lstm_model.h5")

    with open("models/saved/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metrics_data = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "symbol": args.symbol,
    }
    with open("models/saved/metrics.json", "w") as f:
        import json
        json.dump(metrics_data, f, indent=4)

    print("✅ Artefatos salvos em models/saved/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--no_synthetic", action="store_true")
    args = parser.parse_args()
    main(args)
