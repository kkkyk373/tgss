import torch
import torch.nn as nn
import torch.nn.functional as F

# FYI: https://github.com/scikit-mobility/DeepGravity/tree/master/deepgravity/models

# --- 1. DeepGravityモデル（実験済み） ---
class DeepGravityReg(nn.Module):
    """
        MLPReg
        DeepGravityReg: A simple feedforward neural network for gravity model
        input_dim: number of features (e.g., 3 for origin, destination, and distance)
        hidden_dims: list of hidden layer dimensions
        Note that this model does not include softmax normalization.
        This is because the model is used for regression tasks, where the output is a continuous value.
    """
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        flow_pred = self.output_layer(features).squeeze(-1)
        return flow_pred


# --- 2. DeepGravity_tsinghuaモデル（ベンチマークと同一） ---
class DeepGravity_tsinghua(nn.Module):
    """
        from https://github.com/tsinghua-fib-lab/CommutingODGen-Dataset/tree/main/models/DGM
    """
    def __init__(self):
        super(DeepGravity_tsinghua, self).__init__()

        hiddim = 256
        layers = 15

        self.linear_in = nn.Linear(263, hiddim)
        self.linears = nn.ModuleList(
            [nn.Linear(hiddim, hiddim) for i in range(layers)]
        )
        self.linear_out = nn.Linear(hiddim, 1)

    def forward(self, input):
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x)) + x
        x = torch.tanh(self.linear_out(x))
        return x


class OD_normer():
    """
        from https://github.com/scikit-mobility/DeepGravity/tree/master/deepgravity/models
        オリジン・デスティネーションの特徴量を正規化するクラス
        ユースケース: DeepGravity_tsinghua で、オリジン・デスティネーションの特徴量を正規化するために使用
    """
    def __init__(self, min_, max_):
        self.min_ = min_
        self.max_ = max_

    def normalize(self, x):
        """Scale a value or array of values to the range [-1, 1]."""
        return 2 * ((x - self.min_) / (self.max_ - self.min_)) - 1

    def renormalize(self, x):
        return ((x + 1) / 2) * (self.max_ - self.min_) + self.min_


class GravityPower(nn.Module):
    """古典的な重力モデル（べき乗則版）"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 入力xの想定: [origin_pop, dest_pop, distance]
        origin_pop, dest_pop, distance = x[:, 0], x[:, 1], x[:, 2]
        eps = 1e-8
        log_y = (self.alpha * torch.log(origin_pop + eps) + 
                 self.beta * torch.log(dest_pop + eps) - 
                 self.gamma * torch.log(distance + eps))
        return torch.exp(log_y)


class GravityExponential(nn.Module):
    """古典的な重力モデル（指数関数版）"""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # 入力xの想定: [origin_pop, dest_pop, distance]
        origin_pop, dest_pop, distance = x[:, 0], x[:, 1], x[:, 2]
        eps = 1e-8
        log_y = (self.alpha * torch.log(origin_pop + eps) + 
                 self.beta * torch.log(dest_pop + eps) - 
                 self.gamma * distance)
        return torch.exp(log_y)