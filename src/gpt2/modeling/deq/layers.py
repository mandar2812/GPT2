from abc import abstractmethod
import math
from torch import nn


def initialize_nn_weights(tensor, param=0.01):
    return (
        tensor.uniform_()
        * math.sqrt(param)
        / math.sqrt((tensor.shape[0] + tensor.shape[1]))
    )


class _PcDEQLinearLayer(nn.Module):
    def __init__(self, out_features, act):
        super().__init__()
        self.w = nn.Linear(out_features, out_features, bias=False)
        self.w.weight = nn.Parameter(initialize_nn_weights(self.w.weight.data, 1e-4))
        self.w = nn.utils.weight_norm(self.w)
        self.activation = self._get_activation(act)

    def forward(self, z, x):
        z = self.activation(self.w(z) + x)
        return z

    @abstractmethod
    def _get_activation(self, act):
        pass


class PcDEQ1LinearLayer(_PcDEQLinearLayer):
    def _get_activation(self, act):
        match act:
            case "tanh":
                return nn.Tanh()
            case "softsign":
                return nn.Softsign()
            case "relu6":
                return nn.ReLU6()
            case _:
                raise NotImplementedError(
                    f"Activation function '{act}' currently is not supported"
                )


class PcDEQ2LinearLayer(_PcDEQLinearLayer):
    def _get_activation(self, act):
        if act == "sigmoid":
            return nn.Sigmoid()

        raise NotImplementedError(
            f"Activation function '{act}' currently is not supported"
        )


class _PcDEQConvLayer(nn.Module):
    def __init__(self, out_features, act):
        super().__init__()
        self.w = nn.Conv2d(out_features, out_features, 3, padding=1, bias=False)
        self.w.weight = nn.Parameter(initialize_nn_weights(self.w.weight.data, 1e-4))
        self.w = nn.utils.weight_norm(self.w)
        self.activation = self._get_activation(act)

    def forward(self, z, x):
        z = self.activation(self.w(z) + x)
        return z

    @abstractmethod
    def _get_activation(self, act):
        pass


class PcDEQ1ConvLayer(_PcDEQConvLayer):
    def _get_activation(self, act):
        match act:
            case "tanh":
                return nn.Tanh()
            case "softsign":
                return nn.Softsign()
            case "relu6":
                return nn.ReLU6()
            case _:
                raise NotImplementedError(
                    f"Activation function '{act}' currently is not supported"
                )


class PcDEQ2ConvLayer(_PcDEQConvLayer):
    def _get_activation(self, act):
        if act == "sigmoid":
            return nn.Sigmoid()

        raise NotImplementedError(
            f"Activation function '{act}' currently is not supported"
        )


class PcTransformerDEQLayer(nn.Module):
    def __init__(self, d_model, nhead, hidden, activation):
        super().__init__()
        self.d_model = d_model
        self.w_attn = nn.Linear(d_model, d_model, bias=False)
        self.w_ff = nn.Linear(d_model, hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, bias=False)
        self.act = activation

    def forward(self, z0, x, src_mask=None):
        # initialize z with positive values
        z = z0.abs() + 1e-3
        x_pos = x.relu()
        # attention + feedforward, ensure non-negativity
        z_attn, _ = self.attn(z, z, z, attn_mask=src_mask)
        z_attn = self.w_attn(z_attn.relu())  # enforce W ≥ 0 by positive input
        ff = self.act(self.w_ff(z_attn) + x_pos)
        out = self.w_out(ff.relu())
        z = self.act(out)
        return z
