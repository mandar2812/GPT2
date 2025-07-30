from typing import Callable, Protocol
import torch
import torch.nn.utils.parametrize as P


class FixedPointSolver(Protocol):
    def __call__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        z_0: torch.Tensor,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> tuple[torch.Tensor, list[float], int]:
        """Solves the fixed point equation z = f(z) using an iterative method."""


def fixed_point_iteration(
    f: Callable[[torch.Tensor], torch.Tensor], z_0: torch.Tensor, max_iter=200, tol=1e-4
):
    z = z_0
    res = []
    with P.cached():
        for i in range(1, max_iter + 1):
            z_next = f(z)
            err = torch.norm(z_next - z).item() / (torch.norm(z_next).item() + 1e-5)
            z = z_next
            res.append(err)
            if err < tol:
                break

    return z, res, i
