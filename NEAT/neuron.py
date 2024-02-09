from typing import List

import NEAT.activations as act
import NEAT.genome as gen
import NEAT.link as link


class Neuron:
    def __init__(self, neuron_id: int, bias: float, activation: act.Activation):
        self._neuron_id = neuron_id
        self._bias = bias
        self._activation = activation
        self._links: List[link.Link] = []

    def add_input(self, input_id: int, weight: float):
        self._links.append(link.Link(input_id, self._neuron_id, weight))

    def activation_fn(self, x: float):
        return act.ActivationFn(x)(self.activation)

    def activation(self):
        return self._activation

    def id(self):
        return self._neuron_id

    def bias(self):
        return self._bias

    def links(self):
        return self._links
