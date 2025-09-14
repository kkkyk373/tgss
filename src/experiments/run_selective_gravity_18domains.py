import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import random

from src.models.gravity import GravityPower


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_fgw_distances(fgw_dir: str, alpha: float):
    """Load FGW area IDs and distance matrix for given alpha in [0,1]."""
    print(f"[INFO] Loading FGW distances for alpha={alpha}...", flush=True)
    area_ids_path = os.path.join(fgw_dir, "fgw_area_ids.npy")
    code = int(round(alpha * 100))
    dist_mat_path = os.path.join(fgw_dir, f"fgw_dist_{code:02d}.dat")

    area_ids = np.load(area_ids_path)
    dist_mat = np.memmap(dist_mat_path, dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    return area_ids, dist_mat


def _load_area_arrays(data_dir: str, area: str):
    import numpy as np
    import os
    p = os.path.join(data_dir, area)
    F = np.load(os.path.join(p, 'F.npy'))  # (N, F_node)
    C = np.load(os.path.join(p, 'C.npy'))  # (N, N)
    Y = np.load(os.path.join(p, 'Y.npy'))  # (N, N)
    return F, C, Y


def _area_pair_count(data_dir: str, area: str) -> int:
    # Avoid materializing all pairs: read shape of Y.npy
    Y = np.load(os.path.join(data_dir, area, 'Y.npy'))
    N = Y.shape[0]
    return N * N


def extract_xy_18domains(data_dir: str, areas, mass_col: int = 0, max_samples=None, seed=42):
    """
    Build feature/target arrays for given `areas`.
    - If `max_samples` is None, load all pairs per area.
    - Else, sample `max_samples` pairs uniformly across the concatenation of areas
      without materializing all pairs at once.
    Features: [mass_i, mass_j, distance_ij]
    Target:   Y[i, j]
    """
    rng = np.random.default_rng(seed)

    if max_samples is None:
        X_list, y_list = []
        for area in areas:
            F, C, Y = _load_area_arrays(data_dir, area)
            masses = F[:, mass_col]
            N = len(masses)
            # Construct all pairs
            i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            i_idx = i_idx.reshape(-1)
            j_idx = j_idx.reshape(-1)
            x = np.stack([
                masses[i_idx],
                masses[j_idx],
                C[i_idx, j_idx]
            ], axis=1).astype(np.float32)
            y = Y[i_idx, j_idx].astype(np.float32)
            X_list.append(x)
            y_list.append(y)
        if not X_list:
            return np.array([]), np.array([])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    # Sample across all areas
    area_sizes = [ _area_pair_count(data_dir, area) for area in areas ]
    total = sum(area_sizes)
    if total == 0:
        return np.array([]), np.array([])

    if total <= max_samples:
        return extract_xy_18domains(data_dir, areas, mass_col, max_samples=None, seed=seed)

    global_indices = rng.choice(total, size=max_samples, replace=False)
    global_indices.sort()

    X_list, y_list = [], []
    cum = 0
    for area, size in zip(areas, area_sizes):
        if size == 0:
            continue
        start = cum
        end = cum + size
        sel = global_indices[(global_indices >= start) & (global_indices < end)]
        if len(sel) > 0:
            # Map local index -> (i, j)
            F, C, Y = _load_area_arrays(data_dir, area)
            masses = F[:, mass_col]
            N = masses.shape[0]
            local = sel - start
            i_idx = local // N
            j_idx = local % N
            x = np.stack([
                masses[i_idx],
                masses[j_idx],
                C[i_idx, j_idx]
            ], axis=1).astype(np.float32)
            y = Y[i_idx, j_idx].astype(np.float32)
            X_list.append(x)
            y_list.append(y)
        cum = end

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Final shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_and_evaluate_gravity(X_train, y_train, X_test, y_test, target_id, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [INFO] Using device: {device}")

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = GravityPower().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    print(f"    [Train] GravityPower for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        print(f"    Epoch {epoch+1}/{args.epochs} - Loss: {total/len(train_dl.dataset):.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor.to(device)).cpu().numpy()
    mse = float(mean_squared_error(y_test, pred))

    learned = {
        "alpha": float(model.alpha.detach().cpu().item()),
        "beta": float(model.beta.detach().cpu().item()),
        "gamma": float(model.gamma.detach().cpu().item()),
    }

    return mse, learned


def run_all_targets(area_ids, dist_mat, source_ids, args):
    print(f"[INFO] Loading targets from {args.targets_path}")
    with open(args.targets_path) as f:
        targets_raw = [line.strip() for line in f if line.strip()]
    targets = [t for t in targets_raw if t in area_ids]

    # Preload training data when condition == "all"
    X_train_all, y_train_all = None, None
    if args.condition == "all":
        sidx_all = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])
        selected_areas_all = area_ids[sidx_all]
        X_train_all, y_train_all = extract_xy_18domains(
            args.data_dir, selected_areas_all, mass_col=args.mass_col,
            max_samples=args.max_samples, seed=args.seed
        )
        if len(X_train_all) == 0:
            print("[ERROR] No training data for 'all' condition.")
            return []

    results = []
    print(f"[INFO] Evaluating {len(targets)} targets...")
    sidx = np.array([np.where(area_ids == sid)[0][0] for sid in source_ids if sid in area_ids])

    for target in tqdm(targets, desc="Evaluating Targets"):
        print(f"--- Target: {target} ---")
        # Test data for this target (full, no sampling)
        X_test, y_test = extract_xy_18domains(
            args.data_dir, [target], mass_col=args.mass_col,
            max_samples=None, seed=args.seed
        )

        # Select training areas
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
            X_train, y_train = extract_xy_18domains(
                args.data_dir, selected_areas, mass_col=args.mass_col,
                max_samples=args.max_samples, seed=args.seed
            )

        # Train/Evaluate
        if len(X_train) == 0 or len(y_train) == 0:
            status, mse_val, learned = "skipped_no_train_data", None, None
            print("   [WARN] No training data. Skipping.")
        elif len(X_test) == 0 or len(y_test) == 0:
            status, mse_val, learned = "skipped_no_test_data", None, None
            print("   [WARN] No test data. Skipping.")
        else:
            mse_val, learned = train_and_evaluate_gravity(
                X_train, y_train, X_test, y_test, target, args
            )
            status = "success" if not np.isnan(mse_val) else "skipped_nan_mse"

        results.append({
            "target_id": target,
            "mse": mse_val if mse_val is not None else None,
            "test_samples": int(len(y_test)),
            "train_samples": int(len(y_train)) if len(y_train) else 0,
            "status": status,
            "learned_params": learned,
        })

        if status == "success":
            print(f"   -> MSE: {mse_val:.4f} (train_n={len(y_train)}, test_n={len(y_test)})\n")
        else:
            print(f"   -> Skipped: {status}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Selective Transfer with GravityPower on 18domains")
    # Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Path to 18domains root directory')
    parser.add_argument('--fgw_dir', type=str, required=True, help='Path to directory with FGW outputs')
    parser.add_argument('--targets_path', type=str, required=True, help='Path to target area IDs list')
    parser.add_argument('--sources_path', type=str, required=True, help='Path to source area IDs list')
    parser.add_argument('--results_dir', type=str, default='results', help='Where to save final JSON')
    # Selection
    parser.add_argument('--condition', type=str, required=True, choices=['topk', 'bottomk', 'random', 'all'])
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--bottom_k', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5, help='FGW alpha in [0,1]')
    parser.add_argument('--max_samples', type=int, default=20000, help='Samples to cap training set')
    parser.add_argument('--mass_col', type=int, default=0, help='Column in F.npy for mass')
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-3)
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load FGW
    area_ids, dist_mat = load_fgw_distances(args.fgw_dir, args.alpha)
    with open(args.sources_path) as f:
        source_ids = [line.strip() for line in f if line.strip()]

    # Evaluate
    results = run_all_targets(area_ids, dist_mat, source_ids, args)

    # Save
    final_output = {"metadata": vars(args), "results": results}

    import datetime
    final_output["metadata"]["execution_datetime"] = datetime.datetime.now().isoformat()

    save_dir = os.path.join(
        args.results_dir,
        "gravity",
        "raw",
        args.condition,
        f"alpha{int(round(args.alpha*100))}",
        f"seed{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)
    fname = f"ms{args.max_samples}_bs{args.batch_size}_ep{args.epochs}.json"
    path = os.path.join(save_dir, fname)
    with open(path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"[INFO] Saved results to: {path}")


if __name__ == '__main__':
    main()

