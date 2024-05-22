import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from .mapping import InputBasedWeighting, WeightedFunctions
import lib
from collections import OrderedDict


def attenuated_kaiming_uniform_(
    tensor, a=math.sqrt(5), scale=1.0, mode="fan_in", nonlinearity="leaky_relu"
):
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape}")

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None

        attenuated_kaiming_uniform_(self.weight)
        nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        x_some = x_num
        assert x_some is not None
        x1 = self.weight[None] * x_num[:, :, None] + self.bias[None]
        return x1


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        dropout: float,
        init_scale: float = 0.01,
        n_functions=4,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        super().__init__()
        self.n_functions = n_functions
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)

        self.W_out = WeightedFunctions(
            n_functions=self.n_functions, in_dim=d, out_dim=d
        )
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            attenuated_kaiming_uniform_(m.weight, scale=init_scale)
            nn_init.zeros_(m.bias)

        if self.W_out is not None:
            attenuated_kaiming_uniform_(self.W_out.weight)
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def get_attention_mask(self, input_shape, device):
        bs, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = (
            seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        )
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        return attention_mask

    def forward(
        self,
        x: Tensor,
        weights: Tensor,
    ) -> Tensor:
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key)  # b f f
        masks = self.get_attention_mask(attention_scores.shape, attention_scores.device)
        attention = F.softmax(attention_scores + masks, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x, weights[0])
        return x


class XTFormer(nn.Module):
    def __init__(
        self,
        *,
        DataSet_Info: dict,
        token_bias: bool,
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        n_functions: int,
        init_scale: float = 0.1,
    ) -> None:

        super().__init__()

        self.d_token = d_token
        self.dataset_info = DataSet_Info
        self.prenormalization = True

        self.noise = nn.ParameterDict()
        self.noise.name = "noise"
        self.tokenizer = nn.ModuleDict()
        self.tokenizer.name = "tokenizer"
        self.head_fcs = nn.ModuleDict()
        self.head_fcs.name = "head_fcs"
        self.last_fcs = nn.ModuleDict()
        self.last_fcs.name = "last_fcs"

        for dataset_name, dataset_detail in DataSet_Info.items():

            if dataset_name in self.noise.keys():
                dataset_name = "mul_" + dataset_name

            self.noise[dataset_name] = nn.Parameter(
                torch.randn(self.dataset_info[dataset_name]["n_tokens"], 1)
            ).cuda()
            self.tokenizer[dataset_name] = Tokenizer(
                self.dataset_info[dataset_name]["d_numerical"],
                self.dataset_info[dataset_name]["categories"],
                d_token,
                token_bias,
            )
            if dataset_detail["d_out"] == 1:
                self.head_fcs[dataset_name], self.last_fcs[dataset_name] = (
                    self._get_prediction_head(dataset_name)
                )
            else:
                self.head_fcs[dataset_name], self.last_fcs[dataset_name] = (
                    self._get_prediction_head(
                        dataset_name, output_dim=dataset_detail["d_out"]
                    )
                )

        self.mapping = InputBasedWeighting(n_functions=n_functions, n_weights=20)
        self.n_functions = n_functions

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token,
                        n_heads,
                        attention_dropout,
                        init_scale=init_scale,
                        n_functions=self.n_functions,
                    ),
                    "linear0": WeightedFunctions(
                        n_functions=self.n_functions,
                        in_dim=d_token,
                        out_dim=2 * d_token,
                    ),
                    "norm1": make_normalization(),
                }
            )

            attenuated_kaiming_uniform_(layer["linear0"].weight, scale=init_scale)
            nn_init.zeros_(layer["linear0"].bias)
            layer["norm0"] = make_normalization()

            self.layers.append(layer)

        self.activation = lib.get_activation_fn("tanglu")
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def _get_prediction_head(self, dataset_name, output_dim=1) -> nn.Module:
        head = nn.Sequential(
            OrderedDict(
                [
                    ("PReLU", nn.PReLU()),
                    (
                        "n_compression",
                        nn.Linear(
                            self.d_token, self.dataset_info[dataset_name]["d_out"]
                        ),
                    ),
                ]
            )
        )
        last_fc = nn.Linear(self.dataset_info[dataset_name]["n_tokens"], output_dim)

        attenuated_kaiming_uniform_(head.n_compression.weight)
        attenuated_kaiming_uniform_(last_fc.weight, output_dim)

        return head, last_fc

    def use_module(self, module, dataset_name, x, work=True):
        if work:
            return module[dataset_name](x)
        else:
            return module(x)

    def _start_residual(self, x, layer, norm_idx, dataset_name):
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = self.use_module(
                    layer[norm_key], dataset_name, x_residual, work=False
                )
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx, dataset_name):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            norm_key = f"norm{norm_idx}"
            x = self.use_module(layer[norm_key], dataset_name, x, work=False)
        return x

    @property
    def name(self):
        return "xtformer"

    def forward(
        self,
        dataset_name: str,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        mixup: bool = False,
        beta=0.5,
        mtype="feat_mix",
    ) -> Tensor:

        z = self.mapping(self.noise[dataset_name])

        assert x_cat is None
        if mixup and mtype == "naive_mix":
            x_num, feat_masks, shuffled_ids = lib.mixup_data(x_num, beta=beta)
        x = self.tokenizer[dataset_name](x_num)
        if mixup and mtype != "naive_mix":
            mixup_func = {
                "feat_mix": lib.batch_feat_shuffle,
                "hidden_mix": lib.batch_dim_shuffle,
            }[mtype]
            x, feat_masks, shuffled_ids = mixup_func(x, beta=beta)

        for index, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = self._start_residual(x, layer, 0, dataset_name)
            weights = z[index * 2 : 1 + index * 2]
            x_residual = layer["attention"](x_residual, weights)
            x = self._end_residual(x, x_residual, layer, 0, dataset_name)
            x_residual = self._start_residual(x, layer, 1, dataset_name)
            weights = z[index * 2 + 1 : 2 + index * 2]
            x_residual = torch.sum(
                layer["linear0"](x_residual) * weights[:, :, None, :], dim=-1
            )

            x_residual = self.activation(x_residual)
            x = self._end_residual(x, x_residual, layer, 1, dataset_name)

        x = self.last_fcs[dataset_name](x.transpose(1, 2))[:, :, 0]
        x = self.head_fcs[dataset_name](x)
        x = x.squeeze(-1)

        if mixup:
            return x, feat_masks, shuffled_ids
        return x
