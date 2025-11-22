#!/usr/bin/env python3
"""
Project B — Sequence-Level Intrusion Detection
Usage:
    python projectB_intrusion_detection.py --data ./UNSW_NB15.csv

Requirements:
    pip install pandas numpy scikit-learn tensorflow==2.12 matplotlib
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)[:10]} ...")
    return df


def basic_preprocess(df):
    # Drop non-numeric columns that are obviously metadata; encode some categorical if present
    df = df.copy()
    # common textual columns in UNSW-NB15: 'proto', 'service', 'state', 'attack_cat' optionally
    categorical = []
    for c in ['proto', 'service', 'state', 'attack_cat']:
        if c in df.columns:
            categorical.append(c)
    for c in categorical:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    # If label exists but not binary, make binary (0 normal, 1 attack)
    if 'label' in df.columns:
        # some versions use label 0/1 or 'Normal'/'Attack'
        if df['label'].dtype == object:
            df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal','benign','0'] else 1)
        else:
            df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    elif 'attack_cat' in df.columns:
        df['label'] = df['attack_cat'].apply(lambda x: 0 if str(x).lower() in ['normal'] else 1)

    # Drop high-cardinality text columns if present (like srcip, dstip) - we'll use them for grouping if needed
    meta_cols = []
    for c in ['srcip', 'dstip', 'sport', 'dport', 'stime', 'ltime', 'id']:
        if c in df.columns:
            meta_cols.append(c)

    # Select numeric features only for model inputs
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # ensure 'label' is last
    if 'label' in numeric:
        numeric.remove('label')
    X = df[numeric].fillna(0).astype(float)
    y = df['label'].astype(int) if 'label' in df.columns else pd.Series(np.zeros(len(df), dtype=int))
    return X, y, df, meta_cols


def make_sequences(X_df, y_series, seq_len=20, overlap=0.5, grouping_df=None, meta_cols=None):
    """
    Build sequences.
    If grouping_df has 'srcip' and 'dstip' we group by connection pairs and create sequences per group (ordered as in file).
    Otherwise use sliding windows across whole dataset (time-ordered if timestamp exists).
    """
    X = X_df.values
    y = y_series.values
    n_samples, n_features = X.shape
    step = max(1, int(seq_len * (1 - overlap)))

    sequences = []
    seq_labels = []

    # Try grouping by (srcip, dstip) if available
    if meta_cols:
        if 'srcip' in grouping_df.columns and 'dstip' in grouping_df.columns:
            print("Grouping by (srcip, dstip) to form sequences.")
            grouped = grouping_df.groupby(['srcip', 'dstip']).indices
            for (src, dst), idxs in grouped.items():
                idxs = sorted(idxs)
                arr = X[idxs]
                labs = y[idxs]
                # sliding windows within this connection
                for start in range(0, max(1, len(arr) - seq_len + 1), step):
                    seq = arr[start:start + seq_len]
                    if seq.shape[0] == seq_len:
                        sequences.append(seq)
                        seq_labels.append(int(labs[start:start + seq_len].max()))  # if any attack in seq -> attack
    else:
        # sliding global windows
        print("No grouping columns found — using global sliding windows.")
        for start in range(0, n_samples - seq_len + 1, step):
            seq = X[start:start + seq_len]
            sequences.append(seq)
            seq_labels.append(int(y[start:start + seq_len].max()))

    sequences = np.array(sequences)
    seq_labels = np.array(seq_labels)
    print(f"Built {len(sequences)} sequences of length {seq_len} with shape {sequences.shape}")
    return sequences, seq_labels


def build_model(seq_len, n_features, latent_dim=64, conv_filters=[64, 32], kernel_size=3):
    """
    Builds an autoencoder:
    - Encoder: Conv1D layers -> LSTM compress to latent
    - Decoder: RepeatVector -> LSTM -> Conv1DTranspose (via Conv1D)
    """
    inp = layers.Input(shape=(seq_len, n_features))
    x = inp
    # Convolutional encoder (extract local feature patterns along features axis)
    for f in conv_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    # Now temporal encoder
    x = layers.LSTM(latent_dim, activation='tanh', return_sequences=False)(x)
    # Bottleneck
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    # Decoder
    x = layers.RepeatVector(seq_len // (2 ** len(conv_filters)) if (2 ** len(conv_filters)) > 0 else seq_len)(latent)
    # Upsample using LSTM
    x = layers.LSTM(latent_dim, activation='tanh', return_sequences=True)(x)
    # Upsample back to seq_len using UpSampling1D layers to reverse pooling
    for f in reversed(conv_filters):
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding='same', activation='relu')(x)
    # final projection
    out = layers.Conv1D(filters=n_features, kernel_size=1, activation='linear', padding='same')(x)

    model = models.Model(inp, out, name='cnn_lstm_autoencoder')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    model.summary()
    return model


def main(args):
    df = load_data(args.data)
    X_df, y, original_df, meta_cols = basic_preprocess(df)

    # Optional: sort by time if available (try columns 'stime', 'timestamp', 'start_time')
    for tcol in ['stime', 'start_time', 'timestamp', 'time']:
        if tcol in original_df.columns:
            print(f"Sorting by {tcol}")
            idx_sorted = original_df.sort_values(tcol).index
            X_df = X_df.loc[idx_sorted].reset_index(drop=True)
            y = y.loc[idx_sorted].reset_index(drop=True)
            original_df = original_df.loc[idx_sorted].reset_index(drop=True)
            break

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

    # Make sequences
    sequences, seq_labels = make_sequences(X_scaled, y, seq_len=args.seq_len, overlap=args.overlap, grouping_df=original_df, meta_cols=meta_cols)

    # split into train/val/test with only normal sequences in training (unsupervised)
    normal_idx = np.where(seq_labels == 0)[0]
    attack_idx = np.where(seq_labels == 1)[0]

    # Shuffle
    np.random.shuffle(normal_idx)
    train_n = int(len(normal_idx) * 0.7)
    val_n = int(len(normal_idx) * 0.1)

    train_idx = normal_idx[:train_n]
    val_idx = normal_idx[train_n:train_n + val_n]
    test_idx = np.concatenate([normal_idx[train_n + val_n:], attack_idx])

    X_train = sequences[train_idx]
    X_val = sequences[val_idx]
    X_test = sequences[test_idx]
    y_test = seq_labels[test_idx]

    print(f"Train sequences: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} (attacks in test: {y_test.sum()})")

    # Build model
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_model(seq_len, n_features, latent_dim=args.latent_dim, conv_filters=[args.filters, int(args.filters/2)])

    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Train
    history = model.fit(
        X_train, X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, X_val),
        callbacks=cb,
        verbose=2
    )

    # Compute reconstruction errors on validation to set threshold
    recon_val = model.predict(X_val)
    mse_val = np.mean(np.square(recon_val - X_val), axis=(1,2))
    threshold = np.percentile(mse_val, args.threshold_percentile)
    print(f"Using threshold at {args.threshold_percentile} percentile of val MSE: {threshold:.6f}")

    # Evaluate on test set
    recon_test = model.predict(X_test)
    mse_test = np.mean(np.square(recon_test - X_test), axis=(1,2))
    y_pred = (mse_test > threshold).astype(int)

    auc = roc_auc_score(y_test, mse_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    print("=== Evaluation ===")
    print(f"AUC (MSE scores): {auc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Save model and scaler
    os.makedirs('output', exist_ok=True)
    model.save('output/cnn_lstm_autoencoder.h5')
    import joblib
    joblib.dump(scaler, 'output/scaler.save')
    np.save('output/threshold.npy', np.array([threshold]))

    # Plot loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('output/training_loss.png', dpi=150)
    print("Artifacts saved to ./output/")

    # Save a CSV with test results (mse, label, pred)
    res_df = pd.DataFrame({
        'mse': mse_test,
        'label': y_test,
        'pred': y_pred
    })
    res_df.to_csv('output/test_results.csv', index=False)
    print("Test results saved to output/test_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to UNSW_NB15 CSV file')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length (number of flows per sequence)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap fraction between sequences (0..1)')
    parser.add_argument('--latent_dim', type=int, default=64, help='LSTM latent dimension')
    parser.add_argument('--filters', type=int, default=64, help='Base conv filters (will create half for second conv)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--threshold_percentile', type=float, default=95.0, help='Percentile on val MSE to set anomaly threshold')
    args = parser.parse_args()
    main(args)
