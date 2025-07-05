# === run_selective_dgm.py (全ての変更を反映した最終版) ===
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
import datetime
import json


def load_fgw_distances(fgw_dir, alpha):
    """FGW距離データをロードする"""
    print(f"[INFO] Loading FGW distances for alpha={alpha}...", flush=True)
    area_ids_path = os.path.join(fgw_dir, "fgw_area_ids.npy")
    dist_mat_path = os.path.join(fgw_dir, f"fgw_dist_{alpha:02d}.dat")
    
    area_ids = np.load(area_ids_path)
    dist_mat = np.memmap(dist_mat_path, dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def extract_xy(data_dir, areas, max_samples=None, seed=42):
    """
    元のextract_xyと論理的に等価な結果を、メモリ効率良く生成する関数。
    """
    if not max_samples:
        ds = CommutingODPairDataset(data_dir, areas)
        if len(ds) == 0: return np.array([]), np.array([])
        X = np.stack([s["x"] for s in ds])
        y = np.stack([s["y"] for s in ds])
        return X, y

    print(f"    [Data] Starting logically-equivalent sampling from {len(areas)} areas...", flush=True)
    np.random.seed(seed)

    area_sizes = [len(CommutingODPairDataset(data_dir, [area])) for area in areas]
    total_samples = sum(area_sizes)

    if total_samples == 0:
        return np.array([]), np.array([])
    
    if total_samples <= max_samples:
        print(f"    [Data] Total samples ({total_samples}) is less than or equal to max_samples ({max_samples}). Using all data.", flush=True)
        return extract_xy(data_dir, areas, max_samples=None, seed=seed)

    global_indices_to_sample = np.random.choice(total_samples, max_samples, replace=False)
    global_indices_to_sample.sort()

    X_list, y_list = [], []
    cumulative_size = 0

    for area, size in zip(areas, area_sizes):
        if size == 0:
            continue
        
        start_range = cumulative_size
        end_range = cumulative_size + size
        
        indices_in_this_area_range = global_indices_to_sample[
            (global_indices_to_sample >= start_range) & (global_indices_to_sample < end_range)
        ]

        if len(indices_in_this_area_range) > 0:
            local_indices = indices_in_this_area_range - start_range
            
            ds_single = CommutingODPairDataset(data_dir, [area])
            X_area = np.stack([ds_single[i]["x"] for i in local_indices])
            y_area = np.stack([ds_single[i]["y"] for i in local_indices])
            
            X_list.append(X_area)
            y_list.append(y_area)

        cumulative_size += size

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    final_shuffle_idx = np.random.permutation(len(X))
    X = X[final_shuffle_idx]
    y = y[final_shuffle_idx]

    return X, y


def train_and_evaluate_dgm(X_train, y_train, X_test, y_test, args):
    """
    Deep Gravity Modelを学習させ、テストデータで評価する。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [INFO] Using device: {device}", flush=True)

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = DeepGravityReg(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    print(f"    [Train] Starting DGM training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = F.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"    Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_epoch_loss:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(X_test_tensor.to(device))
        pred = pred_tensor.cpu().numpy()

    mse = mean_squared_error(y_test, pred)

    if args.model_output_dir:
        model_save_dir = args.model_output_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"dgm_{now}"
            f"_alpha{args.alpha}"
            f"_{args.condition}"
            f"_seed{args.seed}.pt"
        )
        save_path = os.path.join(model_save_dir, fname)
        
        torch.save(model.state_dict(), save_path)
        print(f"    [INFO] Saved model -> {save_path}", flush=True)

    return mse


def run_single_target(target, area_ids, dist_mat, source_ids, args):
    """単一のターゲット都市に対して、ソース選択、学習、評価を行う"""
    tidx = np.where(area_ids == target)[0][0]
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
    dists = dist_mat[tidx]

    if args.condition == "topk":
        selected_indices = sidx[np.argsort(dists[sidx])[:args.top_k]]
    elif args.condition == "bottomk":
        selected_indices = sidx[np.argsort(-dists[sidx])[:args.bottom_k]]
    elif args.condition == "random":
        selected_indices = np.random.choice(sidx, args.top_k, replace=False)
    elif args.condition == "all":
        selected_indices = sidx
    else:
        raise ValueError(f"Unknown condition: {args.condition}")
    
    selected_areas = area_ids[selected_indices]

    X_train, y_train = extract_xy(args.data_dir, selected_areas, args.max_samples, seed=args.seed)
    X_test, y_test = extract_xy(args.data_dir, [target], max_samples=None, seed=args.seed)

    if len(X_train) == 0:
        print(f"    [WARN] No training data for selected sources. Skipping target {target}.", flush=True)
        return np.nan, len(y_test), 0
    if len(X_test) == 0:
        print(f"    [WARN] No test data for target {target}. Skipping.", flush=True)
        return np.nan, 0, len(y_train)
    
    mse = train_and_evaluate_dgm(X_train, y_train, X_test, y_test, args)
    return mse, len(y_test), len(y_train)


def run_all_targets(area_ids, dist_mat, source_ids, args):
    """全てのターゲット都市に対して評価を実行し、結果のリストを返す"""
    print(f"[INFO] Loading targets from {args.targets_path}", flush=True)
    with open(args.targets_path) as f:
        targets_raw = [line.strip() for line in f if line.strip()]
    targets = [t for t in targets_raw if t in area_ids]

    results_list = []
    print(f"[INFO] Evaluating {len(targets)} targets...", flush=True)

    for target in tqdm(targets, desc="Evaluating Targets"):
        print(f"--- Evaluating target: {target} ---", flush=True)
        try:
            mse, test_n, train_n = run_single_target(target, area_ids, dist_mat, source_ids, args)
            result_item = {
                "target_id": target,
                "mse": mse if not np.isnan(mse) else None,
                "test_samples": test_n,
                "train_samples": train_n,
                "status": "success" if not np.isnan(mse) else "skipped_no_data"
            }
            results_list.append(result_item)
            if not np.isnan(mse):
                print(f"    -> MSE: {mse:.4f} (train_n={train_n}, test_n={test_n})\n", flush=True)
            else:
                print(f"    -> Skipped due to no data.\n", flush=True)
        except Exception as e:
            print(f"    [ERROR] Failed on target {target}: {e}\n", file=sys.stderr, flush=True)
            error_item = {
                "target_id": target, "mse": None, "test_samples": 0,
                "train_samples": 0, "status": "error", "error_message": str(e)
            }
            results_list.append(error_item)
            
    return results_list


def main():
    parser = argparse.ArgumentParser(description="Selective Transfer Learning with Deep Gravity Model")
    # --- Path Arguments ---
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--fgw_dir', type=str, required=True, help="Path to the directory containing FGW distances.")
    parser.add_argument('--targets_path', type=str, required=True, help="Path to the file listing target area IDs.")
    parser.add_argument('--sources_path', type=str, required=True, help="Path to the file listing source area IDs.")
    parser.add_argument('--results_dir', type=str, default='results', help="Directory to save final JSON results.")
    parser.add_argument('--model_output_dir', type=str, default='outputs', help="Directory to save trained models.")
    
    # --- Selection Strategy Arguments ---
    parser.add_argument('--condition', type=str, required=True, choices=['topk', 'bottomk', 'random', 'all'], help="Source selection condition.")
    parser.add_argument('--top_k', type=int, default=100, help="Number of source areas for top-k/random.")
    parser.add_argument('--bottom_k', type=int, default=100, help="Number of source areas for bottom-k.")
    parser.add_argument('--alpha', type=int, default=50, help="Alpha value for FGW distance.")
    parser.add_argument('--max_samples', type=int, default=5000, help="Maximum number of samples to use for training.")
    
    # --- Model Training Arguments ---
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    
    # --- Reproducibility Arguments ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # --- Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # --- Data Loading ---
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    # --- Run Evaluation ---
    evaluation_results = run_all_targets(area_ids, dist_mat, source_ids, args)

    # --- Save Results ---
    final_output = {
        "metadata": vars(args),
        "results": evaluation_results
    }
    final_output["metadata"]["execution_datetime"] = datetime.datetime.now().isoformat()
    
    results_save_dir = os.path.join(
        args.results_dir, "dgm", "raw", 
        args.condition, 
        f"alpha{args.alpha}", 
        f"seed{args.seed}"
    )
    os.makedirs(results_save_dir, exist_ok=True)
    
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"dgm_results_{now_str}.json"
    
    output_path = os.path.join(results_save_dir, fname)

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n[INFO] Successfully saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()