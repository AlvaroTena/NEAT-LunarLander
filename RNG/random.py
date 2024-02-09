from threading import Lock

import numpy as np

lock = Lock()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(
                        *args, **kwargs
                    )
        return cls._instances[cls]


class Rng(metaclass=Singleton):

    def __init__(self, seed_config=None, min_val=0.0, max_val=1.0):
        self.min = min_val
        self.max = max_val
        if seed_config is not None:
            self.set_seed(seed_config)

    def set_seed(self, seed):
        np.random.seed(seed)

    def next_int(self, lower, upper=None):
        if upper is None:
            upper = lower
            lower = 0
        return np.random.randint(lower, upper + 1)

    def next_double(self, lower=0.0, upper=None):
        if upper is None:
            upper = lower
            lower = 0.0
        return lower + np.random.random() * (upper - lower)

    def next_gaussian(self, mean, stddev):
        return np.random.normal(mean, stddev)

    def coin_toss(self, true_probability):
        return self.next_double() <= true_probability

    def choose(self, probability, a, b):
        return a if self.coin_toss(probability) else b

    def choose_3(self, first_probability, second_probability):
        p = self.next_double()
        if p <= first_probability:
            return 0
        elif p <= first_probability + second_probability:
            return 1
        else:
            return 2

    def choose_random(self, samples):
        idx = self.next_int(0, len(samples) - 1)
        return samples[idx]

    def choose_probability(self, samples, probabilities):
        return np.random.choice(len(samples), p=probabilities)
