import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils.dataset import CommutingODPairDataset
import random

def load_fgw_distances(fgw_dir, alpha):
    area_ids = np.load(f"{fgw_dir}/fgw_area_ids.npy")
    N = len(area_ids)
    dist_file = f"{fgw_dir}/fgw_dist_{alpha:02d}.dat"
    dist_mat = np.memmap(dist_file, dtype=np.float32, mode="r", shape=(N, N))
    return area_ids, dist_mat

def extract_xy(data_dir, areas_list, max_samples=None, seed=42):
    ds = CommutingODPairDataset(data_dir, areas_list)
    X = np.stack([s["x"] for s in ds], axis=0)
    y = np.stack([s["y"] for s in ds], axis=0)

    if max_samples is not None and len(X) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    return X, y

def train_and_eval(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, len(X_train)

def run_single_condition(condition, area_ids, dist_mat, args):
    tidxs = np.where(area_ids == args.target_area)[0]
    if len(tidxs) == 0:
        raise ValueError(f"TARGET_AREA `{args.target_area}` not found.")
    tidx = tidxs[0]
    dists = dist_mat[tidx]

    if condition == "topk":
        order = np.argsort(dists)
        selected_idxs = order[order != tidx][:args.top_k]
    elif condition == "bottomk":
        order = np.argsort(dists)[::-1]
        selected_idxs = order[order != tidx][:args.bottom_k]
    elif condition == "random":
        all_idxs = np.delete(np.arange(len(area_ids)), tidx)
        selected_idxs = np.random.choice(all_idxs, args.top_k, replace=False)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    selected_areas = area_ids[selected_idxs].tolist()
    print(f"Condition: {condition}, Selected areas: {selected_areas}")

    X_src, y_src = extract_xy(args.data_dir, selected_areas, max_samples=args.max_samples)
    X_tgt, y_tgt = extract_xy(args.data_dir, [args.target_area], max_samples=args.max_samples)

    print(f"Training on condition: {condition}...")
    mse, size = train_and_eval(X_src, y_src, X_tgt, y_tgt)

    print("\n===== Result =====")
    print(f"Condition: {condition}")
    print(f"Training size: {size} samples -> MSE: {mse:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fgw_dir', type=str, required=True)
    parser.add_argument('--target_area', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True, choices=['topk', 'bottomk', 'random'])
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--bottom_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)
    run_single_condition(args.condition, area_ids, dist_mat, args)

if __name__ == "__main__":
    main()
