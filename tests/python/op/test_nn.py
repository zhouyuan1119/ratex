import copy

import pytest
import torch
import torch.nn as nn

import torch_mnm
from torch_mnm.testing import verify_step


def test_conv():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.conv = nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False
            )

        def forward(self, x):
            out = self.conv(x)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)
    verify_step(Test(), [x])


def test_linear():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.linear = nn.Linear(120, 84)

        def forward(self, x):
            out = self.linear(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Test(), [x])


def test_sum():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, x):
            out = torch.sum(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Test(), [x], jit_script=False)


def test_pad():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, x):
            pad = (1, 2, 3, 4, 5, 6)
            out = torch.nn.functional.pad(x, pad, "constant", 2)
            return out

    shape = [32, 120, 20]
    x = torch.randn(*shape)
    verify_step(Test(), [x], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
