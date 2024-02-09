from typing import List, Union

import NEAT.activations as act
import NEAT.link_gene as lgen
import NEAT.neuron_gene as ngen
import numpy as np
from RNG.random import Rng


class Genome:
    def __init__(self, genome_id: int, num_inputs: int, num_outputs: int):
        self._genome_id = genome_id
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._neurons: List[ngen.NeuronGene] = []
        self._links: List[lgen.LinkGene] = []

    def id(self):
        return self._genome_id

    def add_neuron(self, neuron: ngen.NeuronGene):
        self._neurons.append(neuron)

    def add_link(self, link: lgen.LinkGene):
        assert (
            link.link_id.input_id < 0 or link.link_id.input_id >= self._num_outputs
        ), "Link ID input validation failed."
        self._links.append(link)

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def num_hidden(self):
        # Inputs are not counted as neurons here.
        return len(self._neurons) - self._num_outputs

    def neurons(self):
        return self._neurons

    def links(self):
        return self._links

    def find_neuron(self, neuron_id: int) -> Union[ngen.NeuronGene, None]:
        for neuron in self._neurons:
            if neuron.neuron_id == neuron_id:
                return neuron
        return None

    def get_neuron(self, neuron_id: int) -> ngen.NeuronGene:
        for neuron in self._neurons:
            if neuron.neuron_id == neuron_id:
                return neuron
        raise ValueError(f"Failed to find neuron: {neuron_id}")

    def find_link(self, link_id: lgen.LinkId) -> Union[lgen.LinkGene, None]:
        for link in self._links:
            if link.link_id == link_id:
                return link
        return None

    def get_link(self, link_id: lgen.LinkId) -> lgen.LinkGene:
        for link in self._links:
            if link.link_id == link_id:
                return link
        raise ValueError(
            f"Failed to find link: {link_id.input_id} -> {link_id.output_id}"
        )

    def make_input_ids(self):
        return [-i - 1 for i in range(self._num_inputs)]

    def make_output_ids(self):
        return [i for i in range(self._num_outputs)]
