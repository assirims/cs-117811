# ==================== preprocessing.py ====================
# Min-Max normalization

# preprocessing.py
from sklearn.preprocessing import MinMaxScaler

def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled