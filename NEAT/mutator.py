import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import NEAT.activations as act
import NEAT.config as cfg
import NEAT.genome as gen
import NEAT.indexer as idx
import NEAT.link_gene as lgen
import NEAT.neuron_gene as ngen
from RNG.random import Rng


class MarkState:
    BackEdge = 1
    CrossEdge = 2


def dfs(
    neuron_id: int, edges: Dict[int, List[int]], mark: Dict[int, MarkState]
) -> bool:
    if neuron_id in mark:
        return mark[neuron_id] == MarkState.BackEdge
    mark[neuron_id] = MarkState.BackEdge
    for output_id in edges[neuron_id]:
        if dfs(output_id, edges, mark):
            return True
    mark[neuron_id] = MarkState.CrossEdge
    return False


def would_create_cycle(
    links: List[lgen.LinkGene], new_input_id: int, new_output_id: int
) -> bool:
    mark = {}
    edges = defaultdict(list)
    for link in links:
        edges[link.link_id.input_id].append(link.link_id.output_id)
    edges[new_input_id].append(new_output_id)

    # No need to try running this for new link because if it adds cycle then it
    # would be covered by existing links.
    for link in links:
        if dfs(link.link_id.input_id, edges, mark):
            return True
    return False


class ActivationMutator:
    def __init__(self, config: cfg.ActivationConfig, rng: Rng):
        self._config = config
        self._rng = rng

    def new_value(self):
        return self._config.default_value

    def mutate(self, activation: act.Activation):
        if len(self._config.options) == 0:
            return activation
        if self._rng.coin_toss(self._config.mutation_rate):
            return self._rng.choose_random(self._config.options)
        return activation


class BoolMutator:

    def __init__(self, config: cfg.BoolConfig, rng: Rng):
        self._config = config
        self._rng = rng

    def new_value(self):
        return self._config.default_value

    def mutate(self, value: bool):
        mutation_rate = (
            self._config.true_mutate_rate if value else self._config.false_mutate_rate
        )
        if self._rng.coin_toss(mutation_rate):
            return self._rng.coin_toss(0.5)
        return value


class DoubleMutator:
    def __init__(self, config: cfg.DoubleConfig, rng: Rng):
        self._config = config
        self._rng = rng

    def new_value(self):
        return self._clamp(
            self._rng.next_gaussian(self._config.init_mean, self._config.init_stdev)
        )

    def mutate_delta(self, value: float):
        delta = self._clamp(self._rng.next_gaussian(0.0, self._config.mutate_power))
        return self._clamp(value + delta)

    def mutate(self, value: float):
        choice = self._rng.choose_3(
            self._config.mutation_rate, self._config.replace_rate
        )
        if choice == 0:
            return self.mutate_delta(value)
        elif choice == 1:
            return self.new_value()
        else:
            return value

    def _clamp(self, x: float):
        return min(self._config.max, max(self._config.min, x))


class NeuronMutator:
    def __init__(self, config: cfg.NeuronConfig, rng: Rng, neuron_indexer: idx.Indexer):
        self._config = config
        self._activation_mutator = ActivationMutator(self._config.activation, rng)
        self._bias_mutator = DoubleMutator(self._config.bias, rng)
        self._neuron_indexer = neuron_indexer

    def new_neuron(self, neuron_id_opt: int = None):
        neuron_id = self._next_neuron_id(neuron_id_opt)
        bias = self._bias_mutator.new_value()
        activation = self._activation_mutator.new_value()
        return ngen.NeuronGene(neuron_id, bias, activation)

    def mutate(self, gene: ngen.NeuronGene):
        gene.activation = self._activation_mutator.mutate(gene.activation)
        gene.bias = self._bias_mutator.mutate(gene.bias)
        return gene

    def _next_neuron_id(self, neuron_id_opt: int = None):
        if neuron_id_opt is not None:
            return neuron_id_opt
        return self._neuron_indexer.next()


class LinkMutator:

    def __init__(self, config: cfg.LinkConfig, rng: Rng):
        self._config = config
        self._weight_mutator = DoubleMutator(self._config.weight, rng)
        self._is_enabled_mutator = BoolMutator(self._config.is_enabled, rng)

    def new_value(self, input_id: int, output_id: int):
        weight = self._weight_mutator.new_value()
        is_enabled = self._is_enabled_mutator.new_value()
        return lgen.LinkGene(lgen.LinkId(input_id, output_id), weight, is_enabled)

    def mutate(self, gene: lgen.LinkGene):
        gene.weight = self._weight_mutator.mutate(gene.weight)
        return gene


class GenomeMutator:
    def __init__(
        self,
        config: cfg.GenomeConfig,
        rng: Rng,
        genome_indexer: idx.Indexer,
        neuron_indexer: idx.Indexer,
    ):
        self._config = config
        self._neuron_mutator = NeuronMutator(config.neuron, rng, neuron_indexer)
        self._link_mutator = LinkMutator(config.link, rng)
        self._genome_indexer = genome_indexer
        self._rng = rng

    def new_genome(self):
        genome = gen.Genome(
            self._genome_indexer.next(),
            self._config.num_inputs,
            self._config.num_outputs,
        )
        for i in range(self._config.num_outputs):
            genome.add_neuron(self._neuron_mutator.new_neuron(i))

        # Fully connected direct feed-forward
        for i in range(self._config.num_inputs):
            # We are using negative numbers to represent input nodes.
            input_id = -i - 1
            for output_id in range(self._config.num_outputs):
                genome.add_link(self._link_mutator.new_value(input_id, output_id))

        return genome

    def mutate(self, genome: gen.Genome):
        if self._rng.coin_toss(self._config.neuron.add_rate):
            genome = self._mutate_add_neuron(genome)

        if self._rng.coin_toss(self._config.neuron.remove_rate):
            genome = self._mutate_remove_neuron(genome)

        if self._rng.coin_toss(self._config.link.add_rate):
            genome = self._mutate_add_link(genome)

        if self._rng.coin_toss(self._config.link.remove_rate):
            genome = self._mutate_remove_link(genome)

        for neuron in genome.neurons():
            neuron = self._neuron_mutator.mutate(neuron)

        for link in genome.links():
            link = self._link_mutator.mutate(link)

    def _mutate_add_neuron(self, genome: gen.Genome):
        if len(genome.links()) == 0:
            # Neurons are added by splitting the link.
            # If no links, then we can't add new neuron.
            return genome

        link_to_split: lgen.LinkGene = self._rng.choose_random(genome.links())
        link_to_split.is_enabled = False

        new_neuron = self._neuron_mutator.new_neuron()
        genome.add_neuron(new_neuron)

        genome.add_link(
            lgen.LinkGene(
                lgen.LinkId(
                    link_to_split.link_id.input_id,
                    new_neuron.neuron_id,
                ),
                1.0,
                True,
            )
        )
        genome.add_link(
            lgen.LinkGene(
                lgen.LinkId(new_neuron.neuron_id, link_to_split.link_id.output_id),
                link_to_split.weight,
                True,
            )
        )

        return genome

    def _mutate_remove_neuron(self, genome: gen.Genome):
        if genome.num_hidden() == 0:
            # Nothing to remove
            return genome

        # Choose random hidden neuron
        neurons = genome.neurons()
        neuron_it: ngen.NeuronGene = self._rng.choose_random(
            neurons[genome.num_outputs() :]
        )

        # Delete both neuron and associated links
        links = genome.links()
        genome._links = [
            link
            for link in links
            if link.link_id.input_id != neuron_it.neuron_id
            and link.link_id.output_id != neuron_it.neuron_id
        ]

        genome._neurons = [
            neuron for neuron in neurons if neuron.neuron_id != neuron_it
        ]
        return genome

    def _mutate_add_link(self, genome: gen.Genome):
        # Attempt to add a new connection, the only restriction being that the
        # output node cannot be one of the network input pins.
        input_id = self._choose_random_input(genome)
        output_id = self._rng.choose_random(genome.neurons()).neuron_id
        link_id = lgen.LinkId(input_id, output_id)

        # Don't duplicate links.
        existing_link = genome.find_link(link_id)
        if existing_link is not None:
            # At least enable it
            existing_link.is_enabled = True
            return genome

        # Only supporting feed-forward networks for now.
        if would_create_cycle(genome.links(), input_id, output_id):
            return genome

        new_link = self._link_mutator.new_value(input_id, output_id)
        genome.add_link(new_link)
        return genome

    def _mutate_remove_link(self, genome: gen.Genome):
        if len(genome.links()) == 0:
            # Neurons are added by splitting the link.
            # If no links, then we can't add new neuron.
            return genome

        links = genome.links()
        to_remove_it = self._rng.choose_random(links)
        genome._links = [link for link in links if link.link_id != to_remove_it.link_id]
        return genome

    def _choose_random_input(self, genome: gen.Genome) -> int:
        # A bit hacky to workaround the current encoding of inputs and neurons.
        # Inputs are encoded as negative numbers from [-1, -num_inputs].
        # Hidden nodes are neurons at indexes [num_outputs, end).
        # We want to choose an input uniformly at random, so we
        # compute the probability that it's an input (if not, it's hidden).
        p_input = genome.num_inputs() / genome.num_inputs() + genome.num_hidden()
        if self._rng.coin_toss(p_input):
            return -self._rng.next_int(genome.num_inputs() - 1) - 1
        else:
            neurons = genome.neurons()
            return self._rng.choose_random(neurons[genome.num_outputs() :]).neuron_id
