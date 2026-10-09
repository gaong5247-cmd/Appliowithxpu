"""Compare vectorized RVC segmentation against the original reference loop.
Host CPU tests check numerical/gradient equivalence without claiming XPU speed.
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.algorithm.commons import slice_segments


def old_slice(x, starts, size, dim):
    if dim == 2:
        out = torch.zeros_like(x[:, :size])
    else:
        out = torch.zeros_like(x[:, :, :size])
    for i in range(x.shape[0]):
        j = int(starts[i])
        if dim == 2:
            out[i] = x[i, j:j+size]
        else:
            out[i] = x[i, :, j:j+size]
    return out


def verify(dim):
    shape = (4, 35) if dim == 2 else (4, 3, 35)
    a = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    starts = torch.tensor([0, 2, 7, 20])
    result = slice_segments(a, starts, 8, dim=dim)
    expected = old_slice(b, starts, 8, dim=dim)
    assert torch.equal(result, expected)
    result.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(a.grad, b.grad)
    print(f"PASS dim={dim}: output and gradient match reference")


if __name__ == "__main__":
    verify(2)
    verify(3)
