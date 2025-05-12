import torch
import torch.nn as nn
import torch.nn.functional as F

# 使わない
class DeepGravityEasy(nn.Module):
    """
        DeepGravityEasy: A simple feedforward neural network for gravity model
        input_dim: number of features (e.g., 3 for origin, destination, and distance)
        hidden_dims: list of hidden layer dimensions
    """
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(DeepGravity, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, origin_ids):
        """
            N: number of (i,j) pairs
            x: Tensor of shape (N, input_dim)
            origin_ids: Tensor of shape (N,), indicates origin index of each pair
        """
        # (N, hidden_dim)
        features = self.feature_extractor(x)

        # (N,)
        logits = self.output_layer(features).squeeze(-1)

        # Softmax normalization within each origin group
        output = torch.zeros_like(logits)
        for origin in torch.unique(origin_ids):
            mask = (origin_ids == origin)
            output[mask] = F.softmax(logits[mask], dim=0)

        # predicted P(j|i)
        return output


# 使わない
class DeepGravity(nn.Module):
    """
        DeepGravityEasy を計算量的に効率化したもの
    """
    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]): 
            layers += [nn.Linear(d_in, d_out), nn.ReLU()] 
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, origin_ids):
        """
            x: Tensor of shape (M, input_dim), M = number of (i,j) pairs
            origin_ids: Tensor of shape (M,), indicates origin index of each pair
        """
        logits = self.output_layer(self.feature_extractor(x)).squeeze(-1)  # (M,)

        # --- grouped softmax (originごと) ---
        # ① max for numerical stability
        m = torch.zeros_like(logits)
        m.scatter_reduce_(0, origin_ids, logits, reduce='amax', include_self=False)
        logits_exp = torch.exp(logits - m[origin_ids])

        # ② denominator per origin
        denom = torch.zeros_like(logits_exp)
        denom.scatter_add_(0, origin_ids, logits_exp)

        # ③ softmax
        # logits_exp: (M,)
        return logits_exp / denom[origin_ids]


class DeepGravityReg(nn.Module):
    """
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

        # shape: (N,)
        flow_pred = self.output_layer(features).squeeze(-1)

        # ← softmax なし
        return flow_pred


class GRAVITY_P(nn.Module):
    def __init__(self, input_dim=3):
        assert input_dim == 3, "input_dim must be 3"
        super(GRAVITY_P, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.G = nn.Parameter(torch.tensor([torch.randn(1)]))

    def forward(self, x):
        x = x + 1e-10 # avoid log(0)
        logy = self.alpha * torch.log(x[:, 0]) + self.beta * torch.log(x[:, 1]) - self.gamma * torch.log(x[:, 2])
        y = self.G * torch.exp(logy)
        return y


class GRAVITY_E(nn.Module):
    def __init__(self, input_dim=3):
        assert input_dim == 3, "input_dim must be 3"
        super(GRAVITY_E, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.G = nn.Parameter(torch.tensor([torch.randn(1)]))

    def forward(self, x):
        x = x + 1e-10 # avoid log(0)
        logy = self.alpha * torch.log(x[:, 0]) + self.beta * torch.log(x[:, 1]) - self.gamma * x[:, 2]
        y = self.G * torch.exp(logy)
        return y