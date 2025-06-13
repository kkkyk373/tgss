# === run_selective_svr.py (Optuna + internal data split) ===
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # train_test_splitをインポート
from src.utils.dataset import CommutingODPairDataset
import random
import os
import sys
import optuna # Optunaをインポート
import warnings

# Optunaのワーニングを抑制 (optional)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def load_fgw_distances(fgw_dir, alpha):
    area_ids = np.load(f"{fgw_dir}/fgw_area_ids.npy")
    dist_mat = np.memmap(f"{fgw_dir}/fgw_dist_{alpha:02d}.dat", dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def extract_xy(data_dir, areas, max_samples=None, seed=42):
    ds = CommutingODPairDataset(data_dir, areas)
    X = np.stack([s["x"] for s in ds])
    y = np.stack([s["y"] for s in ds])
    
    # max_samplesが指定されており、かつデータ数がそれより多い場合
    if max_samples and len(X) > max_samples:
        # np.random.seed(seed) は呼び出し元のseedに依存するため、
        # ここでは常に同じサブセットを選ぶようにseedを固定する
        # もしくは、関数外で設定されたseedをargs経由で渡す
        rng = np.random.default_rng(seed) # 最新の推奨される乱数ジェネレータ
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# run_single_target 関数の変更点
def run_single_target(target, area_ids, dist_mat, source_ids, args):
    tidx = np.where(area_ids == target)[0][0]
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
    dists = dist_mat[tidx]

    if args.condition == "topk":
        selected = sidx[np.argsort(dists[sidx])[:args.top_k]]
    elif args.condition == "bottomk":
        selected = sidx[np.argsort(-dists[sidx])[:args.bottom_k]]
    elif args.condition == "random":
        selected = np.random.choice(sidx, args.top_k, replace=False) # np.random.choiceはグローバルseedに依存
    else:
        raise ValueError("Unknown condition")

    # ここで訓練に使用するソースエリアのデータを取得
    # このデータは、SVRのハイパーパラメータチューニングのための内部訓練・検証に用いられる
    source_X, source_y = extract_xy(args.data_dir, area_ids[selected], args.max_samples, seed=args.seed)
    
    # ターゲットエリアのテストデータを取得
    X_test_final, y_test_final = extract_xy(args.data_dir, [target], args.max_samples, seed=args.seed)

    # =========================================================================
    # Optuna を用いたSVRのハイパーパラメータチューニング
    # objective 関数: 最適化したい指標（RMSE）を返す関数を定義
    def objective(trial):
        # 訓練データをさらに内部で訓練用と検証用に分割
        # Optunaの各trial内で一貫した分割を行うため、ここでrandom_stateを固定する
        # ただし、args.seedを直接使うと、親ループのseedと混同するため、
        # trial.suggest_intでランダムなシードを生成することも可能
        # 今回は、args.seedを基点としたランダムなシードを使うか、シンプルに固定値にする
        # ここでは、args.seedを使い、再現性を確保しつつ試行ごとの分割を固定
        try:
            # 訓練データが十分にない場合を考慮し、最低限の分割を試みる
            # train_sizeで訓練データの割合、test_sizeで検証データの割合
            # stratifyはyが分類問題の場合に層化抽出するが、回帰なのでNone
            # random_stateはOptunaのtrialごとに同じになるようにargs.seedを使う
            X_train_inner, X_valid_inner, y_train_inner, y_valid_inner = train_test_split(
                source_X, source_y, test_size=0.2, random_state=args.seed, shuffle=True
            )
        except ValueError as e:
            # train_test_splitが失敗した場合 (例: データ数が少なすぎる)
            print(f"Warning: Not enough data for train_test_split in trial {trial.number}. Error: {e}")
            return float('inf') # これも大きな値を返してペナルティ

        # 探索するハイパーパラメータの範囲を定義
        C = trial.suggest_loguniform('C', 1e-1, 1e3)
        epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e0)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])

        degree = 3
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)

        gamma = 'scale'
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 1e-2, 1e-1, 1e0, 1e1])

        # SVRモデルのインスタンス化
        model = SVR(
            C=C,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            cache_size=200,
            max_iter=10000 # 収束しない場合のために最大イテレーションを設定
        )

        # モデルの訓練 (内部訓練データを使用)
        try:
            model.fit(X_train_inner, y_train_inner)
        except ValueError as e:
            print(f"SVR fit error in trial {trial.number}: {e}")
            return float('inf')

        # 内部検証データでの評価
        y_pred_inner = model.predict(X_valid_inner)
        mse_inner = mean_squared_error(y_valid_inner, y_pred_inner)

        return mse_inner # 最小化したい指標（内部検証MSE）を返す

    # OptunaのStudyを作成し、最適化を実行
    # 各ターゲットエリア/ソースエリア選択組み合わせに対し、個別のStudyを作成
    # samplerのseedは再現性のため
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    # n_trials: 試行回数。計算リソースと相談して決める
    # timeout: 最大実行時間（秒）。
    # verbose=FalseでOptunaの各試行の詳細ログを抑制し、出力を見やすくする
    study.optimize(objective, n_trials=50, timeout=600, show_progress_bar=False) # progress_barはColab/Jupyter向け

    # 最適なパラメータを持つSVRモデルを最終的に構築
    best_params = study.best_params
    
    # 注意: ここではsource_X, source_y (元の選ばれたソースエリアデータ全体) で最終学習
    # もし、訓練・検証分割後のX_train_innerで学習したい場合はロジックを調整
    # 一般的には、チューニングで最適なパラメータを見つけたら、
    # そのパラメータで「利用可能な全ての訓練データ」を使って最終モデルを学習します。
    final_model = SVR(
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        kernel=best_params['kernel'],
        degree=best_params.get('degree', 3),
        gamma=best_params.get('gamma', 'scale'),
        cache_size=200,
        max_iter=10000
    )
    final_model.fit(source_X, source_y) # 選ばれたソースエリアデータ全体で最終学習

    # 最終的なターゲットエリアのテストデータでの評価
    pred_final = final_model.predict(X_test_final)
    mse_final = mean_squared_error(y_test_final, pred_final)
    # =========================================================================

    # mse_finalを返すように変更
    return mse_final, len(y_test_final), len(source_X) # 訓練データのサイズはsource_Xのサイズとする


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
            # run_single_targetは既にOptunaによるチューニングと評価を含む
            mse, test_n, train_n = run_single_target(target, area_ids, dist_mat, source_ids, args)
            total_mse += mse * test_n
            total_test += test_n
            print(f"MSE: {mse:.4f} (train={train_n}, test={test_n})\n", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed on {target}: {e}", flush=True)
            # エラーが発生したターゲットについては、計算から除外するか、
            # もしくは大きなエラー値を加算するなどの対応を検討
            # ここでは単にスキップ
            continue # エラーのターゲットはスキップ

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
    parser.add_argument('--condition', required=True, choices=['topk', 'bottomk', 'random'])
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--bottom_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42) # シード引数を追加
    args = parser.parse_args()

    # 全体の乱数シードを設定
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    print("[INFO] Loading FGW distances...", flush=True)
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)

    print(f"[INFO] Loading source area IDs from {args.sources_path}", flush=True)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    print("[INFO] Starting evaluation...", flush=True)
    run_all_targets(area_ids, dist_mat, source_ids, args)


if __name__ == "__main__":
    main()