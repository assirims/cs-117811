# ==================== train.py ====================
# Orchestrates the full AIBID-SCSA pipeline

# train.py
from data_loader import load_toniot_dataset
from issa import ISSAFeatureSelector
from roa import ROAOptimizer
from utils import train_and_evaluate
from config import (DATA_DIR, ISSA_POP_SIZE, ISSA_MAX_ITER,
                    ROA_POP_SIZE, ROA_MAX_ITER, HYPERPARAM_RANGES)


def main():
    # 1. Load data
    X_df, y = load_toniot_dataset(DATA_DIR)
    X = X_df.values

    # 2. Feature Selection (ISSA)
    fs = ISSAFeatureSelector(ISSA_POP_SIZE, ISSA_MAX_ITER, chaotic=CHAOTIC_INIT)
    mask = fs.select(X, y)

    # 3. Hyperparameter tuning (ROA)
    def train_fn(lr, batch_size):
        # returns validation accuracy for given params
        history = train_and_evaluate(X, y, mask, {'learning_rate': lr, 'batch_size': batch_size})
        return history.history['val_accuracy'][-1]

    optimizer = ROAOptimizer(ROA_POP_SIZE, ROA_MAX_ITER, HYPERPARAM_RANGES)
    best_params = optimizer.optimize(train_fn)
    print("Best hyperparameters:", best_params)

    # 4. Final training
    train_and_evaluate(X, y, mask, best_params)

if __name__ == '__main__':
    main()