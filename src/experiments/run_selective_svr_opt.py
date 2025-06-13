"""
    Support Vector Regression (SVR) example with Optuna for hyperparameter tuning.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset
from src.utils.split_areas import load_all_areas, split_train_valid_test

import optuna # Optunaをインポート
import warnings

# Optunaのワーニングを抑制する場合 (optional)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# ---------- 1. Data Preparation (unchanged from your sample) ----------
data_dir = "/Users/hideki-h/Desktop/実験用データ/ComOD-dataset/data"
areas = load_all_areas(dir_path=data_dir, if_shuffle=False)
train_areas, valid_areas, _ = split_train_valid_test(
    areas,
    train_ratio=0.1, valid_ratio=0.1, test_ratio=0.1,
    seed=42 # シード値を追加して再現性を確保
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
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")


# ---------- 2. Optuna Objective Function ----------
# Optunaが最適化を行う目的関数を定義します。
# trial: Optunaが提供するオブジェクトで、ハイパーパラメータの試行値を提案します。
def objective(trial):
    # SVRのハイパーパラメータをOptunaに提案させる
    # C: 正則化パラメータ。対数スケールで0.1から1000までを探索。
    C = trial.suggest_loguniform('C', 1e-1, 1e3)
    # epsilon: マージン内の許容誤差。対数スケールで0.01から1までを探索。
    epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e0)
    # kernel: カーネルの種類。'rbf', 'linear', 'poly', 'sigmoid' から選択。
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])

    # kernelが'poly'の場合のみdegree（次数）を探索
    degree = 3 # デフォルト値。もしpoly以外で使われることがないなら影響なし
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5) # 2から5の整数

    # gamma: カーネル係数。'rbf', 'poly', 'sigmoid'の場合に探索。
    # 'scale'と'auto'も選択肢に入れる
    gamma = 'scale' # デフォルト値
    if kernel in ['rbf', 'poly', 'sigmoid']:
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 1e-2, 1e-1, 1e0, 1e1])


    # SVRモデルのインスタンス化
    model = SVR(
        C=C,
        epsilon=epsilon,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        cache_size=200, # キャッシュサイズは計算効率に影響
        max_iter=10000, # 収束しない場合の最大イテレーション数を設定
                       # これを設定しないと、収束しない場合に無限ループになる可能性
    )

    # モデルの学習
    # SVRのfitが収束しないなど、エラーを発生する可能性があるためtry-exceptで囲む
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        # SVRが収束しないなどのエラーが出た場合、大きな値を返すことでペナルティを与える
        print(f"Trial {trial.number} failed to fit with params: {trial.params}. Error: {e}")
        return float('inf') # RMSEを最小化したいので、エラー時は非常に大きな値を返す

    # 検証データでの予測
    y_pred = model.predict(X_valid)

    # 評価指標 (MSE) の計算
    mse = mean_squared_error(y_valid, y_pred)

    return mse # Optunaはこれを最小化しようとします

# ---------- 3. Optuna Study Creation and Optimization ----------
# OptunaのStudyを作成
# direction='minimize' は、目的関数（objectiveが返す値）を最小化することを目指すことを示します。
print("Creating Optuna study...")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# 最適化の実行
# n_trials: 試行するハイパーパラメータの組み合わせの数
# timeout: 最適化を停止するまでの最大時間（秒）
print("Starting Optuna optimization...")
study.optimize(objective, n_trials=50, timeout=300) # 例: 50回の試行、最大5分

# ---------- 4. Best Model Evaluation ----------
print("\nOptimization finished.")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial:")
trial = study.best_trial

print(f"  Value (MSE): {trial.value:.6f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 最適なハイパーパラメータでSVRモデルを再構築し、最終評価
print("\nTraining final SVR model with best parameters...")
final_model = SVR(
    C=trial.params['C'],
    epsilon=trial.params['epsilon'],
    kernel=trial.params['kernel'],
    degree=trial.params.get('degree', 3), # 'degree'はpolyカーネルの時のみ存在
    gamma=trial.params.get('gamma', 'scale'), # 'gamma'は線形カーネルの時など存在しない場合がある
    cache_size=200,
    max_iter=10000
)
final_model.fit(X_train, y_train) # 訓練データ全体で学習

# 最終モデルでの予測と評価 (検証データを使用)
final_y_pred = final_model.predict(X_valid)
final_mse = mean_squared_error(y_valid, final_y_pred)
print(f"\nFinal Model Validation Mean Squared Error (with best params): {final_mse:.6f}")