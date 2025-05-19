# ==================== model.py ====================
# MIX_LSTM-based classification

# model.py
import tensorflow as tf

from config import LSTM_UNITS, DROPOUT_RATE, FC_UNITS, NUM_CLASSES, INPUT_DIM


def build_mix_lstm(input_dim, num_classes, lstm_units=LSTM_UNITS,
                    dropout_rate=DROPOUT_RATE, fc_units=FC_UNITS):
    inputs = tf.keras.Input(shape=(None, input_dim))
    x = inputs
    # Stack Bi-LSTM layers
    for units in lstm_units:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True)
        )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(fc_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model