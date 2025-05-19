# ==================== config.py ====================
# Configuration settings and hyperparameters

# config.py

# Path to pre-downloaded TON_IoT dataset directory
DATA_DIR = "/path/to/TON_IoT_Train_Test_Network"  # adjust accordingly

# General settings
TEST_SIZE = 0.3  # 30% test split
RANDOM_STATE = 42

# ISSA parameters (Feature Selection)
ISSA_POP_SIZE = 30
ISSA_MAX_ITER = 50
CHAOTIC_INIT = True

# MIX_LSTM model parameters
LSTM_UNITS = [64, 32]  # two BiLSTM layers
DROPOUT_RATE = 0.3
FC_UNITS = 64
NUM_CLASSES = 10
INPUT_DIM = None  # to be set after feature selection

# ROA parameters (Hyperparameter Tuning)
ROA_POP_SIZE = 20
ROA_MAX_ITER = 30
HYPERPARAM_RANGES = {
    'learning_rate': (1e-4, 1e-2),
    'batch_size': (32, 128),
}

# Training
EPOCHS = 30
