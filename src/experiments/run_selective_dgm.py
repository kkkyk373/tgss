# === run_selective_dgm.py ===
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset
from src.models.gravity import DeepGravityReg
from tqdm import tqdm
import random
import os
import sys


def load_fgw_distances(fgw_dir, alpha):
    """FGW距離データをロードする"""
    print("[INFO] Loading FGW distances...", flush=True)
    area_ids = np.load(f"{fgw_dir}/fgw_area_ids.npy")
    # memmapで巨大な距離行列を効率的に読み込む
    dist_mat = np.memmap(f"{fgw_dir}/fgw_dist_{alpha:02d}.dat", dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def extract_xy(data_dir, areas, max_samples=None, seed=42):
    """指定されたエリアのデータセットから特徴量(X)とターゲット(y)を抽出する"""
    ds = CommutingODPairDataset(data_dir, areas)
    if len(ds) == 0:
        return np.array([]), np.array([])
    X = np.stack([s["x"] for s in ds])
    y = np.stack([s["y"] for s in ds])
    if max_samples and len(X) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def train_and_evaluate_dgm(X_train, y_train, X_test, y_test, args):
    """
    Deep Gravity Modelを学習させ、テストデータで評価する。
    PyTorchの学習ループを内包する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    print("Debug messages --------- Before converting to tensors ---------")
    # NumPy配列をPyTorchテンソルに変換
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # DataLoaderの作成
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # モデル、オプティマイザの初期化
    input_dim = X_train.shape[1]
    model = DeepGravityReg(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 学習ループ
    model.train()
    print(f"    [Train] Starting DGM training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = F.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            epoch_samples += x_batch.size(0)
        
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        print(f"    Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_epoch_loss:.6f}", flush=True)


    # 評価
    model.eval()
    with torch.no_grad():
        pred_tensor = model(X_test_tensor.to(device))
        pred = pred_tensor.cpu().numpy()

    mse = mean_squared_error(y_test, pred)
    return mse


def run_single_target(target, area_ids, dist_mat, source_ids, args):
    """単一のターゲット都市に対して、ソース選択、学習、評価を行う"""
    tidx = np.where(area_ids == target)[0][0]
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
    dists = dist_mat[tidx]

    # 条件に基づいてソースエリアを選択
    if args.condition == "topk":
        selected_indices = sidx[np.argsort(dists[sidx])[:args.top_k]]
    elif args.condition == "bottomk":
        selected_indices = sidx[np.argsort(-dists[sidx])[:args.bottom_k]]
    elif args.condition == "random":
        selected_indices = np.random.choice(sidx, args.top_k, replace=False)
    else:
        raise ValueError(f"Unknown condition: {args.condition}")
    
    selected_areas = area_ids[selected_indices]

    print("debug messages --------- After selecting sources ---------")

    # 学習データとテストデータを抽出
    X_train, y_train = extract_xy(args.data_dir, selected_areas, args.max_samples, seed=args.seed)
    print("debug messages --------- After extracting training data ---------")
    X_test, y_test = extract_xy(args.data_dir, [target], args.max_samples, seed=args.seed)
    print("debug messages --------- After extracting test data ---------")

    if len(X_train) == 0:
        print(f"    [WARN] No training data for selected sources. Skipping.", flush=True)
        return np.nan, len(y_test), 0
    if len(X_test) == 0:
        print(f"    [WARN] No test data for target {target}. Skipping.", flush=True)
        return np.nan, 0, len(y_train)
    
    print("debug messages --------- After extracting data Before training ---------")

    # Deep Gravity Modelの学習と評価
    mse = train_and_evaluate_dgm(X_train, y_train, X_test, y_test, args)
    
    return mse, len(y_test), len(y_train)


def run_all_targets(area_ids, dist_mat, source_ids, args):
    """全てのターゲット都市に対して評価を実行する"""
    print("[INFO] Entered run_all_targets", flush=True)
    print(f"[INFO] Loading targets from {args.targets_path}", flush=True)

    with open(args.targets_path) as f:
        targets_raw = [line.strip() for line in f if line.strip()]
    targets = [t for t in targets_raw if t in area_ids]

    total_mse, total_test_samples = 0, 0
    print(f"[INFO] Evaluating {len(targets)} targets...", flush=True)

    for i, target in enumerate(tqdm(targets, desc="Evaluating Targets")):
        print(f"[{i+1}/{len(targets)}] Evaluating target: {target}", flush=True)
        try:
            mse, test_n, train_n = run_single_target(target, area_ids, dist_mat, source_ids, args)
            if not np.isnan(mse):
                total_mse += mse * test_n
                total_test_samples += test_n
                print(f"    MSE: {mse:.4f} (train={train_n}, test={test_n})\n", flush=True)
            else:
                print(f"    Skipped due to no data.\n", flush=True)

        except Exception as e:
            print(f"[ERROR] Failed on target {target}: {e}", file=sys.stderr, flush=True)

    if total_test_samples > 0:
        overall_mse = total_mse / total_test_samples
        print(f"\n[RESULT] Overall MSE: {overall_mse:.4f} over {total_test_samples} samples", flush=True)
    else:
        print("[WARNING] No evaluation was performed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Selective Transfer Learning with Deep Gravity Model")
    # --- データ関連の引数 ---
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--fgw_dir', type=str, required=True, help="Path to the directory containing FGW distances.")
    parser.add_argument('--targets_path', type=str, required=True, help="Path to the file listing target area IDs.")
    parser.add_argument('--sources_path', type=str, required=True, help="Path to the file listing source area IDs.")
    
    # --- 選択戦略の引数 ---
    parser.add_argument('--condition', type=str, required=True, choices=['topk', 'bottomk', 'random'], help="Source selection condition.")
    parser.add_argument('--top_k', type=int, default=100, help="Number of source areas for top-k.")
    parser.add_argument('--bottom_k', type=int, default=100, help="Number of source areas for bottom-k.")
    parser.add_argument('--alpha', type=int, default=50, help="Alpha value for FGW distance.")
    parser.add_argument('--max_samples', type=int, default=50000, help="Maximum number of samples to use for training.")
    
    # --- モデル学習の引数 ---
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    
    # --- 再現性の引数 ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # シード値の設定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # データのロード
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    # 評価の実行
    run_all_targets(area_ids, dist_mat, source_ids, args)

if __name__ == "__main__":
    main()
