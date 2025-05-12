import os
import argparse
import numpy as np
import torch
import ot
from tqdm import tqdm

from utils.dataset import CommutingODDataset  # PYTHONPATH=src を仮定


def node_features(x_tensor: torch.Tensor):
    twoFplus1 = x_tensor.shape[-1]
    F = (twoFplus1 - 1) // 2
    return x_tensor[:, 0, :F].numpy()


def dis_path(data_dir: str, aid: str):
    return os.path.join(data_dir, aid, "dis.npy")


def fgw_dist(sample_i, sample_j, data_dir, alpha=0.5):
    x1, aid1 = sample_i["x"], sample_i["area"]
    x2, aid2 = sample_j["x"], sample_j["area"]

    f1 = node_features(x1)
    f2 = node_features(x2)

    C1 = np.load(dis_path(data_dir, aid1)).astype(np.float32)
    C2 = np.load(dis_path(data_dir, aid2)).astype(np.float32)
    C1 /= C1.max() or 1.0
    C2 /= C2.max() or 1.0

    M = np.linalg.norm(f1[:, None, :] - f2[None, :, :], axis=-1)
    p = np.full(f1.shape[0], 1 / f1.shape[0])
    q = np.full(f2.shape[0], 1 / f2.shape[0])

    return ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p=p, q=q,
        loss_fun="square_loss", alpha=alpha, symmetric=True
    )


def main(args):
    os.makedirs(os.path.dirname(args.ids_bin), exist_ok=True)
    area_ids = sorted([
        d for d in os.listdir(args.data_dir)
        if not d.startswith('.') and os.path.isdir(os.path.join(args.data_dir, d))
    ])[:args.n_graphs]

    np.save(args.ids_bin, np.array(area_ids))

    dataset = CommutingODDataset(args.data_dir, area_ids)
    N = len(dataset)

    D = np.memmap(args.dist_bin, mode="w+", dtype=np.float32, shape=(N, N))
    D[:] = 0.0

    for i in tqdm(range(N), desc="FGW rows"):
        samp_i = dataset[i]
        for j in range(i + 1, N):
            samp_j = dataset[j]
            D[i, j] = D[j, i] = fgw_dist(samp_i, samp_j, args.data_dir, args.alpha)

    D.flush()
    print(f"distance matrix  : {args.dist_bin}")
    print(f"area id sequence : {args.ids_bin}")
    print("FGW distance matrix saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing area folders")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha parameter for FGW")
    parser.add_argument("--n_graphs", type=int, default=100,
                        help="Number of graphs to use")
    parser.add_argument("--ids_bin", type=str, default="outputs/fgw_area_ids.npy")
    parser.add_argument("--dist_bin", type=str, default="outputs/fgw_dist.dat")

    args = parser.parse_args()
    main(args)
