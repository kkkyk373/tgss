import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset
import random
import os
import sys


def load_fgw_distances(fgw_dir, alpha):
    area_ids = np.load(f"{fgw_dir}/fgw_area_ids.npy")
    dist_mat = np.memmap(f"{fgw_dir}/fgw_dist_{alpha:02d}.dat", dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def extract_xy(data_dir, areas, max_samples=None, seed=42):
    ds = CommutingODPairDataset(data_dir, areas)
    X = np.stack([s["x"] for s in ds])
    y = np.stack([s["y"] for s in ds])
    if max_samples and len(X) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_single_target(target, area_ids, dist_mat, source_ids, args):
    tidx = np.where(area_ids == target)[0][0]
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
    dists = dist_mat[tidx]

    if args.condition == "topk":
        selected = sidx[np.argsort(dists[sidx])[:args.top_k]]
    elif args.condition == "bottomk":
        selected = sidx[np.argsort(-dists[sidx])[:args.bottom_k]]
    elif args.condition == "random":
        selected = np.random.choice(sidx, args.top_k, replace=False)
    else:
        raise ValueError("Unknown condition")

    X_train, y_train = extract_xy(args.data_dir, area_ids[selected], args.max_samples)
    X_test, y_test = extract_xy(args.data_dir, [target], args.max_samples)

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return mse, len(y_test), len(y_train)


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
    args = parser.parse_args()

    print("[INFO] Loading FGW distances...", flush=True)
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)

    print(f"[INFO] Loading source area IDs from {args.sources_path}", flush=True)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    print("[INFO] Starting evaluation...", flush=True)
    run_all_targets(area_ids, dist_mat, source_ids, args)


if __name__ == "__main__":
    main()
