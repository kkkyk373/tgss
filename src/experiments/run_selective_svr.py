# === run_selective_svr.py (modified) ===
import argparse
import numpy as np
import json
import datetime
import os
import random
import sys
import joblib
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset

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
    if max_samples is None:
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


def train_and_evaluate_svr(X_train, y_train, X_test, y_test, target_id, args):
    """SVRモデルを学習させ、評価し、モデルを保存する関数。"""
    print(f"    [Train] Starting SVR training...", flush=True)
    model = SVR() 
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)

    if args.model_output_dir:
        model_save_dir = os.path.join(
            args.model_output_dir, "svr", args.condition,
            f"alpha{args.alpha}", f"seed{args.seed}"
        )
        os.makedirs(model_save_dir, exist_ok=True)
        
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = f"topk{args.top_k}_ms{args.max_samples}"
        fname = f"svr_target{target_id}_{param_str}_{now}.joblib"
        save_path = os.path.join(model_save_dir, fname)
        
        joblib.dump(model, save_path)
        print(f"    [INFO] Saved model -> {save_path}", flush=True)

    return mse

def run_all_targets(area_ids, dist_mat, source_ids, args):
    """全てのターゲット都市に対して評価を実行し、結果をリストで返す。"""
    print(f"[INFO] Loading targets from {args.targets_path}", flush=True)
    with open(args.targets_path) as f:
        targets_raw = [line.strip() for line in f if line.strip()]
    targets = [t for t in targets_raw if t in area_ids]

    X_train_all, y_train_all = None, None
    if args.condition == "all":
        print("[INFO] Condition is 'all'. Pre-loading training data once...", flush=True)
        sidx_all = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
        selected_areas_all = area_ids[sidx_all]
        X_train_all, y_train_all = extract_xy(args.data_dir, selected_areas_all, args.max_samples, seed=args.seed)
        if len(X_train_all) == 0:
            print("[ERROR] Pre-loading failed for 'all' condition. Aborting.", file=sys.stderr, flush=True)
            return []

    results_list = []
    print(f"[INFO] Evaluating {len(targets)} targets...", flush=True)
    
    # sidxの計算をループの外に移動（最適化）
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])

    for target in tqdm(targets, desc="Evaluating Targets"):
        print(f"--- Evaluating target: {target} ---", flush=True)
        try:
            X_test, y_test = extract_xy(args.data_dir, [target], max_samples=None, seed=args.seed)

            if args.condition == "all":
                X_train, y_train = X_train_all, y_train_all
            else:
                tidx = np.where(area_ids == target)[0][0]
                dists = dist_mat[tidx, sidx]

                if args.condition == "topk":
                    selected_indices = sidx[np.argsort(dists)[:args.top_k]]
                elif args.condition == "bottomk":
                    selected_indices = sidx[np.argsort(-dists)[:args.bottom_k]]
                elif args.condition == "random":
                    selected_indices = np.random.choice(sidx, args.top_k, replace=False)
                else:
                    raise ValueError(f"Unknown condition: {args.condition}")
                
                selected_areas = area_ids[selected_indices]
                X_train, y_train = extract_xy(args.data_dir, selected_areas, args.max_samples, seed=args.seed)

            if len(X_train) == 0:
                status, mse_val = "skipped_no_train_data", None
            elif len(X_test) == 0:
                status, mse_val = "skipped_no_test_data", None
            else:
                mse_val = train_and_evaluate_svr(X_train, y_train, X_test, y_test, target, args)
                status = "success" if not np.isnan(mse_val) else "skipped_nan_mse"

            result_item = {
                "target_id": target, "mse": mse_val, "test_samples": len(y_test),
                "train_samples": len(y_train), "status": status
            }
            results_list.append(result_item)
            
            if status == "success":
                print(f"    -> MSE: {mse_val:.4f}\n", flush=True)
            else:
                print(f"    -> Skipped: {status}\n", flush=True)

        except Exception as e:
            print(f"    [ERROR] Failed on target {target}: {e}\n", file=sys.stderr, flush=True)
            results_list.append({
                "target_id": target, "mse": None, "test_samples": 0,
                "train_samples": 0, "status": "error", "error_message": str(e)
            })
            
    return results_list

def main():
    parser = argparse.ArgumentParser(description="Selective Transfer Learning with SVR")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fgw_dir', type=str, required=True)
    parser.add_argument('--targets_path', type=str, required=True)
    parser.add_argument('--sources_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_output_dir', type=str, default='outputs')
    parser.add_argument('--condition', type=str, required=True, choices=['topk', 'bottomk', 'random', 'all'])
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--bottom_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    evaluation_results = run_all_targets(area_ids, dist_mat, source_ids, args)

    final_output = {
        "metadata": vars(args),
        "results": evaluation_results,
        "execution_datetime": datetime.datetime.now().isoformat()
    }
    
    results_save_dir = os.path.join(
        args.results_dir, "svr", "raw",
        args.condition,
        f"alpha{args.alpha}",
        f"seed{args.seed}"
    )
    os.makedirs(results_save_dir, exist_ok=True)
    
    param_str = (
        f"topk{args.top_k}"
        f"_ms{args.max_samples}"
    )
    fname = f"{param_str}.json"
    
    output_path = os.path.join(results_save_dir, fname)

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"\n[INFO] Successfully saved evaluation results to: {output_path}")

if __name__ == "__main__":
    main()