import torch
from d2l import torch as d2l
import numpy


if __name__ == "__main__":
    device = 'cpu'
    a = torch.randn(size=(1000, 1000), device=device)
    b = torch.mm(a, a)
    with d2l.Benchmark('numpy'):
        for _ in range(10):
            a = numpy.random.normal(size=(1000, 1000))
            b = numpy.dot(a, a)
    with d2l.Benchmark('torch'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)
