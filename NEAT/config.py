from typing import List

import NEAT.activations as act
import numpy as np


class ActivationConfig:
    def __init__(self):
        self.default_value: act.Activation = act.ThresholdFn()
        self.mutation_rate = 0.3
        self.options: List[act.Activation] = [
            act.SigmoidFn(),
            act.ReluFn(),
            act.ThresholdFn(),
        ]


class BoolConfig:
    def __init__(self):
        self.default_value = True
        self.true_mutate_rate = 0.01
        self.false_mutate_rate = 0.01


class DoubleConfig:
    def __init__(
        self,
        init_mean=0.0,
        init_stdev=1.0,
        min_val=-30.0,
        max_val=30.0,
        mutation_rate=0.8,
        mutate_power=0.2,
        replace_rate=0.05,
    ):
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.min = min_val
        self.max = max_val
        self.mutation_rate = mutation_rate
        self.mutate_power = mutate_power
        self.replace_rate = replace_rate


class CompatibilityConfig:
    def __init__(self):
        self.threshold = 2.8
        self.disjoint_coefficient = 1.0
        self.weight_coefficient = 0.65


class NeuronConfig:
    def __init__(self):
        self.add_rate = 0.4
        self.remove_rate = 0.001
        self.activation = ActivationConfig()
        self.bias = DoubleConfig()


class LinkConfig:
    def __init__(self):
        self.add_rate = 0.6
        self.remove_rate = 0.01
        self.weight = DoubleConfig(
            init_mean=0.0,
            init_stdev=1.0,
            min_val=-20.0,
            max_val=20.0,
            mutation_rate=0.3,
            mutate_power=0.5,
            replace_rate=0.1,
        )
        self.is_enabled = BoolConfig()


class GenomeConfig:
    def __init__(self, num_inputs=0, num_outputs=0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neuron = NeuronConfig()
        self.link = LinkConfig()


class ReproductionConfig:
    def __init__(self):
        self.elitism = 30
        self.min_species_size = 1
        self.survival_threshold = 0.15


class StagnationConfig:
    def __init__(self):
        self.elitism = 1
        self.max_stagnation = 100


class NeatConfig:
    def __init__(self, population_size: int, reset_on_extinction: bool):
        self.population_size = population_size
        self.reset_on_extinction = reset_on_extinction
        self.genome = GenomeConfig()
        self.reproduction = ReproductionConfig()
        self.stagnation = StagnationConfig()
        self.compatibility = CompatibilityConfig()
