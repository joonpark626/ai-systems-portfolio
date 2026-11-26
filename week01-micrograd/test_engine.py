import math
import torch
from engine import Value


def test_composite():
    a = Value(2.0, label='a'); b = Value(-3.0, label='b')
    c = Value(10.0, label='c'); f = Value(-2.0, label='f')
    L = (a*b + c) * f
    L.backward()

    at = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    ct = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    ft = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    Lt = (at*bt + ct) * ft
    Lt.backward()

    assert abs(L.data - Lt.item()) < 1e-6
    assert abs(a.grad - at.grad.item()) < 1e-6
    assert abs(b.grad - bt.grad.item()) < 1e-6
    assert abs(c.grad - ct.grad.item()) < 1e-6
    assert abs(f.grad - ft.grad.item()) < 1e-6
