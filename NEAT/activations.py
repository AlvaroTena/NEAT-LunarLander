from typing import Union

import numpy as np


class EmptyFn:
    def __call__(self, x: float) -> float:
        return x


class SigmoidFn:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x / self.p))


class ThresholdFn:
    def __call__(self, x: float) -> float:
        return float(x >= 0.0)


class ReluFn:
    def __call__(self, x: float) -> float:
        return max(0.0, x)


Activation = Union[EmptyFn, SigmoidFn, ReluFn, ThresholdFn]


class ActivationFn:
    def __init__(self, x: float):
        self.x = x

    def __call__(self, fn) -> float:
        return fn(self.x)
