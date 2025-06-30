# === run_selective_svr.py (可読性重視 + 完全なバグ修正版) ===
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # データスケーリング用

from src.utils.dataset import CommutingODPairDataset
import random
import os
import sys
import optuna
import warnings

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def load_fgw_distances(fgw_dir, alpha):
    area_ids = np.load(f"{fgw_dir}/fgw_area_ids.npy")
    dist_mat = np.memmap(f"{fgw_dir}/fgw_dist_{alpha:02d}.dat", dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def extract_xy(data_dir, areas, max_samples=None, seed=42):
    ds = CommutingODPairDataset(data_dir, areas)
    X = np.stack([s["x"] for s in ds])
    y = np.stack([s["y"] for s in ds])
    
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_single_target(target, area_ids, dist_mat, source_ids, args):
    tidx = np.where(area_ids == target)[0][0]
    sidx_all_sources = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])

    # 訓練に使用するソースエリアのID（文字列）のリストを格納する共通変数
    # これが必ず定義されるようにする
    final_selected_area_ids_for_training_list = [] 

    if args.condition == "topk":
        # dist_mat は main 関数で常にロードされるので、None チェックは不要だが、堅牢性のため残してもよい
        # dists の計算は topk/bottomk の場合のみ行う
        dists = dist_mat[tidx] 
        selected_indices_from_sidx_all_sources = sidx_all_sources[np.argsort(dists[sidx_all_sources])[:args.top_k]]
        final_selected_area_ids_for_training_list = area_ids[selected_indices_from_sidx_all_sources]

    elif args.condition == "bottomk":
        # dists の計算は topk/bottomk の場合のみ行う
        dists = dist_mat[tidx]
        selected_indices_from_sidx_all_sources = sidx_all_sources[np.argsort(-dists[sidx_all_sources])[:args.bottom_k]]
        final_selected_area_ids_for_training_list = area_ids[selected_indices_from_sidx_all_sources]

    elif args.condition == "random":
        final_selected_area_ids_for_training_list = np.random.choice(area_ids[sidx_all_sources], args.top_k, replace=False)

    elif args.condition == "all": # 全てのソースエリアのODペアデータを使うベースライン
        final_selected_area_ids_for_training_list = area_ids[sidx_all_sources] 
        
    else:
        raise ValueError("Unknown condition")

    source_X, source_y = extract_xy(args.data_dir, final_selected_area_ids_for_training_list, args.max_samples, seed=args.seed)
    
    X_test_final, y_test_final = extract_xy(args.data_dir, [target], args.max_samples, seed=args.seed)

    # === データスケーリングのセットアップ ===
    scaler_X = StandardScaler()
    source_X_scaled = scaler_X.fit_transform(source_X)

    # =========================================================================
    # Optuna を用いたSVRのハイパーパラメータチューニング
    def objective(trial):
        try:
            X_train_inner_scaled, X_valid_inner_scaled, y_train_inner, y_valid_inner = train_test_split(
                source_X_scaled, source_y, test_size=0.2, random_state=args.seed, shuffle=True
            )
        except ValueError as e:
            print(f"Warning: Not enough data for train_test_split in trial {trial.number}. Error: {e}")
            return float('inf')

        C = trial.suggest_loguniform('C', 1e-1, 1e3)
        epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e0)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])

        degree = 3
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)

        gamma = 'scale'
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 1e-2, 1e-1, 1e0, 1e1])

        model = SVR(
            C=C, epsilon=epsilon, kernel=kernel,
            degree=degree, gamma=gamma,
            cache_size=200, max_iter=10000
        )

        try:
            model.fit(X_train_inner_scaled, y_train_inner)
        except ValueError as e:
            print(f"SVR fit error in trial {trial.number}: {e}")
            return float('inf')

        y_pred_inner = model.predict(X_valid_inner_scaled)
        mse_inner = mean_squared_error(y_valid_inner, y_pred_inner)

        return mse_inner

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.optuna_n_trials, timeout=args.optuna_timeout, show_progress_bar=False)

    best_params = study.best_params
    
    final_model = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        kernel=best_params['kernel'],
        degree=best_params.get('degree', 3),
        gamma=best_params.get('gamma', 'scale'),
        cache_size=200,
        max_iter=10000
    )
    final_model.fit(source_X_scaled, source_y)

    X_test_final_scaled = scaler_X.transform(X_test_final) 
    pred_final = final_model.predict(X_test_final_scaled)
    mse_final = mean_squared_error(y_test_final, pred_final)

    return mse_final, len(y_test_final), len(source_X)

def run_all_targets(area_ids, dist_mat, source_ids, args):
    print("[INFO] Entered run_all_targets", flush=True)
    print(f"[INFO] Loading targets from {args.targets_path}", flush=True)

    with open(args.targets_path) as f:
        targets_raw = [line.strip() for line in f if line.strip()]
    targets = [t for t in targets_raw if t in area_ids]

    total_mse, total_test = 0, 0
    print(f"[INFO] Evaluating {len(targets)} targets...", flush=True)

    for i, target in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {target}", flush=True)
        try:
            mse, test_n, train_n = run_single_target(target, area_ids, dist_mat, source_ids, args)
            total_mse += mse * test_n
            total_test += test_n
            print(f"MSE: {mse:.4f} (train={train_n}, test={test_n})\n", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed on {target}: {e}", flush=True)
            continue

    if total_test:
        print(f"\n[RESULT] Overall MSE: {total_mse / total_test:.4f} over {total_test} samples", flush=True)
    else:
        print("[WARNING] No evaluation performed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--fgw_dir', required=True)
    parser.add_argument('--targets_path', required=True)
    parser.add_argument('--sources_path', required=True)
    parser.add_argument('--condition', required=True, choices=['topk', 'bottomk', 'random', 'all'])
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--bottom_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optuna_n_trials', type=int, default=50)
    parser.add_argument('--optuna_timeout', type=int, default=600)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # main 関数で dist_mat を常にロードする形（可読性重視）
    print("[INFO] Loading FGW distances...", flush=True)
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)

    print(f"[INFO] Loading source area IDs from {args.sources_path}", flush=True)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]
    
    print("[INFO] Starting evaluation...", flush=True)
    run_all_targets(area_ids, dist_mat, source_ids, args)


if __name__ == "__main__":
    main()