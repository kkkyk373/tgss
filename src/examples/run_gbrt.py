"""
    Gradient Boosting Regression Trees (GBRT) example
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset
from src.utils.split_areas import load_all_areas, split_train_valid_test

# ---------- 1. Data ----------
data_dir = "/Users/hideki-h/Desktop/実験用データ/ComOD-dataset/data"
areas = load_all_areas(dir_path=data_dir, if_shuffle=False)
train_areas, valid_areas, _ = split_train_valid_test(
    areas,
    train_ratio=0.1, valid_ratio=0.1, test_ratio=0.1
)

# データセットの準備
train_dataset = CommutingODPairDataset(data_dir, train_areas)
valid_dataset = CommutingODPairDataset(data_dir, valid_areas)

# 特徴量とターゲットを抽出
def extract_features_and_targets(dataset):
    X = np.array([sample["x"] for sample in dataset])  # 特徴量
    y = np.array([sample["y"] for sample in dataset])  # ターゲット
    return X, y

X_train, y_train = extract_features_and_targets(train_dataset)
X_valid, y_valid = extract_features_and_targets(valid_dataset)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ---------- 2. Model Training ----------
# モデル構築
model = GradientBoostingRegressor(
    n_estimators=100,  # 決定木の数
    learning_rate=0.1,  # 学習率
    max_depth=3,  # 各木の最大深さ
    random_state=42
)

# 学習
print("Training the Gradient Boosting Regressor...")
model.fit(X_train, y_train)

# ---------- 3. Evaluation ----------
# 予測
y_pred = model.predict(X_valid)

# 評価
mse = mean_squared_error(y_valid, y_pred)
print(f"Validation Mean Squared Error: {mse:.6f}")