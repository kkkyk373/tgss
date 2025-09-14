import os
import argparse
import numpy as np
import ot
from tqdm import tqdm
from src.utils.dataset import Domains18Dataset


def fgw_distance(F1: np.ndarray, C1: np.ndarray, F2: np.ndarray, C2: np.ndarray, alpha: float) -> float:
    # Normalize intra-graph costs to [0,1] to stabilize FGW
    C1 = C1.astype(np.float32)
    C2 = C2.astype(np.float32)
    max1 = np.max(C1)
    max2 = np.max(C2)
    if max1 > 0:
        C1 = C1 / max1
    if max2 > 0:
        C2 = C2 / max2

    # Feature cost matrix (Euclidean)
    M = np.linalg.norm(F1[:, None, :] - F2[None, :, :], axis=-1).astype(np.float32)

    # Uniform distributions
    n1 = F1.shape[0]
    n2 = F2.shape[0]
    p = np.full(n1, 1.0 / n1, dtype=np.float32)
    q = np.full(n2, 1.0 / n2, dtype=np.float32)

    dist2 = ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p=p, q=q, loss_fun="square_loss", alpha=alpha, symmetric=True
    )
    return float(dist2)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect areas
    areas = [
        d for d in os.listdir(args.data_dir)
        if not d.startswith('.') and os.path.isdir(os.path.join(args.data_dir, d))
    ]
    areas = sorted(areas)
    if args.n_graphs is not None:
        areas = areas[: args.n_graphs]

    # Save area order
    area_ids_path = os.path.join(args.output_dir, "fgw_area_ids.npy")
    np.save(area_ids_path, np.array(areas))

    dataset = Domains18Dataset(args.data_dir, areas)
    N = len(dataset)

    # Compute for each alpha
    for alpha in args.alphas:
        dist_path = os.path.join(args.output_dir, f"fgw_dist_{int(alpha*100):02d}.dat")
        D = np.memmap(dist_path, mode="w+", dtype=np.float32, shape=(N, N))
        D[:] = 0.0

        # Preload all graphs to avoid repeated disk IO
        graphs = []
        for k in range(N):
            item = dataset[k]
            graphs.append((item["F"], item["C"]))

        for i in tqdm(range(N), desc=f"FGW rows (alpha={alpha:.2f})"):
            F1, C1 = graphs[i]
            for j in range(i + 1, N):
                F2, C2 = graphs[j]
                D[i, j] = D[j, i] = fgw_distance(F1, C1, F2, C2, alpha)

        D.flush()
        print(f"[OK] Saved FGW matrix for alpha={alpha:.2f} -> {dist_path}")

    print(f"[OK] Saved area IDs -> {area_ids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FGW distance matrices for 18domains")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to 18domains root directory")
    parser.add_argument("--output_dir", type=str, default="outputs/18domains", help="Directory to save outputs")
    parser.add_argument("--n_graphs", type=int, default=None, help="Limit number of domains (optional)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="List of alpha values")
    args = parser.parse_args()
    main(args)

