from typing import Dict

import NEAT.config as cfg
import NEAT.genome as gen
import NEAT.link_gene as lgen
import NEAT.neuron_gene as ngen


class CacheKey:
    def __init__(self, g1=0, g2=0):
        self.g1, self.g2 = sorted([g1, g2])

    def __hash__(self):
        return hash((self.g1, self.g2))

    def __eq__(self, other):
        return (self.g1, self.g2) == (other.g1, other.g2)


class GenomicDistanceEvaluator:
    def __init__(self, config: cfg.CompatibilityConfig):
        self._config = config
        self._cache: Dict[CacheKey, float] = {}
        self._hits = 0
        self._misses = 0

    def compute(self, g1: gen.Genome, g2: gen.Genome):
        key = CacheKey(g1.id(), g2.id())
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        distance = self._compute_neuron_distances(
            g1, g2
        ) + self._compute_link_distances(g1, g2)
        self._cache[key] = distance
        return distance

    def hits(self):
        return self._hits

    def misses(self):
        return self._misses

    def cache_size(self):
        return len(self._cache)

    def clear_cache(self):
        return self._cache.clear()

    def _compute_neuron_distances(self, g1: gen.Genome, g2: gen.Genome):
        num_disjoint = 0
        matching_distance = 0.0

        g1_neurons = {neuron.neuron_id: neuron for neuron in g1.neurons()}
        g2_neurons = {neuron.neuron_id: neuron for neuron in g2.neurons()}

        for neuron_id, neuron in g1_neurons.items():
            if neuron_id in g2_neurons:
                matching_distance += self._compute_neuron_distance(
                    neuron, g2_neurons[neuron_id]
                )
            else:
                num_disjoint += 1

        for neuron_id in g2_neurons:
            if neuron_id not in g1_neurons:
                num_disjoint += 1

        total_distance = (
            matching_distance + num_disjoint * self._config.disjoint_coefficient
        )
        return total_distance / max(len(g1.neurons()), len(g2.neurons()))

    def _compute_neuron_distance(
        self, neuron1: gen.ngen.NeuronGene, neuron2: gen.ngen.NeuronGene
    ):
        distance = abs(neuron1.bias - neuron2.bias)
        if type(neuron1.activation) is not type(neuron2.activation):
            distance += 1.0
        return distance * self._config.weight_coefficient

    def _compute_link_distances(self, g1: gen.Genome, g2: gen.Genome) -> float:
        matching_distance = 0.0
        num_disjoint = 0

        g1_links = {link.link_id: link for link in g1.links()}
        g2_links = {link.link_id: link for link in g2.links()}

        for link_id, link in g1_links.items():
            if link_id in g2_links:
                matching_distance += self._compute_link_distance(
                    link, g2_links[link_id]
                )
            else:
                num_disjoint += 1

        for link_id in g2_links:
            if link_id not in g1_links:
                num_disjoint += 1

        total_distance = (
            matching_distance + num_disjoint * self._config.disjoint_coefficient
        )
        return total_distance / max(len(g1.links()), len(g2.links()))

    def _compute_link_distance(self, link1: lgen.LinkGene, link2: lgen.LinkGene):
        distance = abs(link1.weight - link2.weight)
        if link1.is_enabled != link2.is_enabled:
            distance += 1.0
        return distance * self._config.weight_coefficient
