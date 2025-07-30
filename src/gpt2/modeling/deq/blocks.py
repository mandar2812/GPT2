import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn
from src.gpt2.modeling.deq.layers import (
    PcDEQ1LinearLayer,
    PcDEQ2LinearLayer,
    PcDEQ1ConvLayer,
    PcDEQ2ConvLayer,
    PcTransformerDEQLayer,
)
from src.gpt2.modeling.deq.solvers import FixedPointSolver, fixed_point_iteration


class DEQFixedPoint(nn.Module):
    def __init__(self, f: nn.Module, solver: FixedPointSolver, **kwargs):
        super().__init__()
        self.f: nn.Module = f
        self.solver = solver
        self.iter_forward = 0
        self.iter_backward = 0
        self.kwargs = kwargs
        self.res_forward = 0
        self.res_backward = 0

    def forward(self, x, **fkwargs):
        with torch.no_grad():
            z, self.res_forward, self.iter_forward = self.solver(
                lambda z: self.f(z, x, **fkwargs), torch.zeros_like(x), **self.kwargs
            )
        z = self.f(z, x, **fkwargs)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x, **fkwargs)

        def backward_hook(grad):
            g, self.res_backward, self.iter_backward = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return g

        z.register_hook(backward_hook)
        return z


class LinearPcDEQ1Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ1LinearLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.softplus(x, beta=5)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.w.weight_v.data.clamp_(min=0)
        self.deq.f.w.weight_g.data.clamp_(min=0)


class LinearPcDEQ2Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ2LinearLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.relu(x)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.w.weight_v.data.clamp_(min=0)
        self.deq.f.w.weight_g.data.clamp_(min=0)


class ConvPcDEQ1Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ1ConvLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.softplus(x, beta=5)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.w.weight_v.data.clamp_(min=0)
        self.deq.f.w.weight_g.data.clamp_(min=0)


class ConvPcDEQ2Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ2ConvLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.relu(x)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.w.weight_v.data.clamp_(min=0)
        self.deq.f.w.weight_g.data.clamp_(min=0)


class PcTransformerDEQBlock(nn.Module):
    def __init__(self, d_model, nhead, hidden, activation, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcTransformerDEQLayer(d_model, nhead, hidden, activation),
            fixed_point_iteration,
            **kwargs
        )

    def forward(self, x, src_mask=None):
        z = self.deq(x, src_mask=src_mask)
        return z

    def clamp(self):
        self.deq.f.w_attn.weight.data.clamp_(min=0)
        self.deq.f.w_ff.weight.data.clamp_(min=0)
        self.deq.f.w_out.weight.data.clamp_(min=0)
