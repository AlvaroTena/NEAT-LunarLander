from collections import defaultdict
from typing import List, Set

import NEAT.activations as act
import NEAT.genome as gen
import NEAT.link_gene as lgen
import NEAT.neuron_gene as ngen
import numpy as np


def required_for_output(
    inputs: List[int], outputs: List[int], links: List[lgen.LinkGene]
):
    inputs_set = set(inputs)
    required = set(outputs)

    s = required.copy()
    found = True
    while found:
        found = False
        for link in links:
            if link.link_id.input_id not in s and link.link_id.output_id in s:
                if link.link_id.input_id not in inputs_set:
                    required.add(link.link_id.input_id)
                s.add(link.link_id.input_id)
                found = True

    return required


def all_inputs_are_activated(
    activated: Set[int], candidate: int, links: List[lgen.LinkGene]
):
    for link in links:
        if (
            candidate == link.link_id.output_id
            and link.link_id.input_id not in activated
        ):
            return False
    return True


def feed_forward_layers(
    inputs: List[int], outputs: List[int], links: List[lgen.LinkGene]
) -> List[int]:
    required = required_for_output(inputs, outputs, links)

    layers = []
    activated = set(inputs)

    while True:
        next_layer = []
        for link in links:
            if (
                link.link_id.input_id in activated
                and link.link_id.output_id not in activated
            ):
                candidate = link.link_id.output_id
                if candidate in required and all_inputs_are_activated(
                    activated, candidate, links
                ):
                    if candidate not in next_layer:
                        next_layer.append(candidate)

        if not next_layer:
            break

        activated.update(next_layer)
        layers.append(next_layer)

    return layers


class NeuronInput:
    def __init__(self, input_id, weight):
        self.input_id = input_id
        self.weight = weight


class NeuronEval:
    def __init__(self, neuron, inputs):
        self.neuron = neuron
        self.inputs = inputs


class FeedForwardNeuralNetwork:
    def __init__(self, inputs: List[int], outputs: List[int], evals: List[NeuronEval]):
        self._inputs = inputs
        self._outputs = outputs
        self._evals = evals
        self._values = defaultdict(float)

    @staticmethod
    def create_from_genome(genome: gen.Genome):
        inputs = genome.make_input_ids()
        outputs = genome.make_output_ids()
        layers = feed_forward_layers(inputs, outputs, genome.links())

        evals = []
        for layer in layers:
            for neuron_id in layer:
                neuron_inputs = [
                    NeuronInput(link.link_id.input_id, link.weight)
                    for link in genome.links()
                    if neuron_id == link.link_id.output_id
                ]
                neuron = genome.find_neuron(neuron_id)
                assert neuron is not None, "Neuron must exist"
                evals.append(NeuronEval(neuron, neuron_inputs))

        return FeedForwardNeuralNetwork(inputs, outputs, evals)

    def activate(self, inputs):
        assert len(inputs) == len(self._inputs), "Input size must match"

        self._values.clear()
        for i, input_value in enumerate(inputs):
            self._values[-i - 1] = input_value

        for eval in self._evals:
            value = (
                sum(
                    self._values[input.input_id] * input.weight for input in eval.inputs
                )
                + eval.neuron.bias
            )
            value = act.ActivationFn(value)(eval.neuron.activation)
            self._values[eval.neuron.neuron_id] = value

        return [self._values[output_id] for output_id in self._outputs]
