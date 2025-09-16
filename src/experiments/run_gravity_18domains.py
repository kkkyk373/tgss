import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

from src.utils.dataset import Domains18PairDataset
from src.models.gravity import GravityPower


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_gravity(model: torch.nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, lr: float, device):
    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-8

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred_log = model(xb)
            target_log = torch.log(yb + eps)
            loss = F.mse_loss(pred_log, target_log)
            loss.backward()
            opt.step()


def evaluate_gravity(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test).float().to(device)
        pred = model.predict_flow(X_tensor).cpu().numpy()
    mse = mean_squared_error(y_test, pred)
    return mse


def main():
    parser = argparse.ArgumentParser(description="Train/Test GravityPower on 18domains")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to 18domains root directory')
    parser.add_argument('--mass_col', type=int, default=0, help='Which column in F.npy to use as mass')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test split ratio within each domain')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='results/gravity/18domains', help='Where to save JSON results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enumerate areas
    areas = [
        d for d in os.listdir(args.data_dir)
        if not d.startswith('.') and os.path.isdir(os.path.join(args.data_dir, d))
    ]
    areas = sorted(areas)

    # Load full pairwise dataset once per-area in loop to limit memory
    results = []
    for area in areas:
        ds = Domains18PairDataset(args.data_dir, areas=[area], mass_col=args.mass_col)
        if len(ds) == 0:
            results.append({"area": area, "status": "empty", "mse": None})
            continue

        X = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
        y = np.stack([float(ds[i]["y"]) for i in range(len(ds))])

        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(n * (1.0 - args.test_ratio))
        train_idx, test_idx = idx[:split], idx[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = GravityPower()
        train_gravity(model, X_train, y_train, args.epochs, args.batch_size, args.lr, device)
        mse = evaluate_gravity(model, X_test, y_test, device)

        item = {
            "area": area,
            "mse": float(mse),
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "alpha": float(model.alpha.detach().cpu().item()),
            "beta": float(model.beta.detach().cpu().item()),
            "gamma": float(model.gamma.detach().cpu().item()),
            "status": "success",
        }
        results.append(item)
        print(f"[OK] {area}: MSE={item['mse']:.4f} (alpha={item['alpha']:.3f}, beta={item['beta']:.3f}, gamma={item['gamma']:.3f})")

    # Save summary
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"gravity_power_seed{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "metadata": {
                "seed": args.seed,
                "test_ratio": args.test_ratio,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "mass_col": args.mass_col,
            },
            "results": results,
        }, f, indent=2)
    print(f"[OK] Saved results -> {out_path}")


if __name__ == '__main__':
    main()
