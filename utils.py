# ==================== utils.py ====================
# Utility functions: train/evaluate

# utils.py
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import TEST_SIZE, RANDOM_STATE, EPOCHS
from preprocessing import normalize_data

import numpy as np


def train_and_evaluate(X, y, feature_mask, hyperparams):
    # Split
    X_sel = X[:, feature_mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Normalize
    X_train, X_test = normalize_data(X_train, X_test)
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train = X_train.reshape((-1, 1, X_train.shape[1]))
    X_test = X_test.reshape((-1, 1, X_test.shape[1]))

    # Build model
    from model import build_mix_lstm
    model = build_mix_lstm(X_train.shape[-1], len(np.unique(y_train)))
    model.compile(
        optimizer=Adam(hyperparams['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=hyperparams['batch_size']
    )
    # Evaluate
    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    return history
