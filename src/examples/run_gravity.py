import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.dataset import CommutingODPairDataset
from src.utils.split_areas import load_all_areas, split_train_valid_test
from src.models.gravity import DeepGravityReg
from tqdm import tqdm

# ---------- 1. Data ----------
data_dir = "/Users/hideki-h/Desktop/実験用データ/ComOD-dataset/data"
areas = load_all_areas(dir_path=data_dir, if_shuffle=False)
train_areas, valid_areas, _ = split_train_valid_test(
    areas,
    train_ratio=0.1, valid_ratio=0.1, test_ratio=0.1
)

train_loader = DataLoader(
    CommutingODPairDataset(data_dir, train_areas),
    batch_size=32, shuffle=False, num_workers=0
)

valid_loader = DataLoader(
    CommutingODPairDataset(data_dir, valid_areas),
    batch_size=32, shuffle=False, num_workers=0
)


# ---------- 2. Train Loop ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル構築
input_dim = train_loader.dataset[0]["x"].shape[-1]
model = DeepGravityReg(input_dim=input_dim).to(device)

# オプティマイザ
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# エポック数
num_epochs = 10

# 学習ループ
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0.0
    total_count = 0

    # tqdm で進捗表示
    for batch in tqdm(train_loader, desc="Train"):
        x = batch["x"].to(device)           # shape: (N, input_dim)
        print(f"x.shape: {x.shape}")
        y_true = batch["y"].to(device)      # shape: (N,)

        # フォワード
        y_pred = model(x)                   # shape: (N,)

        # 損失計算（回帰）
        loss = F.mse_loss(y_pred, y_true)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ロス積算
        total_loss += loss.item() * y_true.size(0)
        total_count += y_true.size(0)

    avg_loss = total_loss / total_count
    print(f"  Avg Train Loss: {avg_loss:.6f}")
