import math
from functools import partial

import torch
import torch.nn as nn
from entmax import entmax15
from torch.nn import init as nn_init


class InputBasedWeighting(nn.Module):
    def __init__(self, n_functions, n_weights, sparsity=False) -> None:
        super(InputBasedWeighting, self).__init__()

        self.mapping = nn.Sequential(
            nn.Linear(1, n_functions * n_weights),
            nn.Softplus(),
            nn.Linear(n_functions * n_weights, n_functions * n_weights),
        )

        if sparsity:
            self.activation = partial(entmax15, dim=-1)
        else:
            self.activation = nn.Softmax(dim=-1)

        self.n_functions = n_functions
        self.n_weights = n_weights

    def forward(self, x):
        x = self.mapping(x)
        x = x.view(-1, self.n_weights, self.n_functions)
        x = self.activation(x).transpose(0, 1)
        return x


class WeightedFunctions(nn.Module):
    def __init__(self, n_functions, in_dim, out_dim, if_bias=True) -> None:
        super(WeightedFunctions, self).__init__()
        self.n_functions = n_functions
        self.functions = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=if_bias) for _ in range(n_functions)]
        )
        self.if_bias = if_bias
        self.init_weight()
        self.weights = None

    @property
    def weight(self):
        return torch.stack(
            [self.functions[i].weight for i in range(self.n_functions)], dim=-1
        )

    @property
    def bias(self):
        if self.if_bias is False:
            return None
        return torch.stack(
            [self.functions[i].bias for i in range(self.n_functions)], dim=-1
        )

    def init_weight(self):
        for i in range(self.n_functions):
            nn_init.kaiming_uniform_(self.functions[i].weight, a=math.sqrt(5))
            if self.if_bias is not None:
                nn_init.zeros_(self.functions[i].bias)

    def forward(self, x, weights=None):
        out = [self.functions[i](x) for i in range(self.n_functions)]
        out = torch.stack(out, dim=-1)

        if weights is None:
            return out
        else:
            out = torch.sum(out * weights[None, :, None, :], dim=-1)
            return out

    def merge_forward(self, x, weights):
        if self.weights is None:
            self.weights = weights
        w = (self.weight[None, :, :, :] * weights[:, None, None, :]).sum(-1)
        b = (self.bias[None, :, :] * weights[:, None, :]).sum(-1)
        out = torch.einsum("bfi,foi->bfo", x, w) + b[None, ...]
        return out
