import NEAT.genome as gen
import NEAT.indexer as idx
import NEAT.individual as ind
import NEAT.link_gene as lgen
import NEAT.neuron_gene as ngen
from RNG.random import Rng


class IndividualCrossover:
    def __init__(self, genome_indexer: idx.Indexer, rng: Rng):
        self._genome_indexer = genome_indexer
        self._rng = rng

    def crossover(self, p1: ind.Individual, p2: ind.Individual):
        if p1.fitness > p2.fitness:
            return self._crossover_internal(p1, p2)
        return self._crossover_internal(p2, p1)

    def _crossover_internal(self, dominant: ind.Individual, recessive: ind.Individual):
        offspring = gen.Genome(
            self._genome_indexer.next(),
            dominant.genome.num_inputs(),
            dominant.genome.num_outputs(),
        )

        # Inherit neuron genes
        for dominant_neuron in dominant.genome.neurons():
            neuron_id = dominant_neuron.neuron_id
            recessive_neuron = recessive.genome.find_neuron(neuron_id)
            if recessive_neuron is None:
                offspring.add_neuron(dominant_neuron)
            else:
                offspring.add_neuron(
                    self._crossover_neuron(dominant_neuron, recessive_neuron)
                )

        # Inherit link genes
        for dominant_link in dominant.genome.links():
            link_id = dominant_link.link_id
            recessive_link = recessive.genome.find_link(link_id)
            if recessive_link is None:
                offspring.add_link(dominant_link)
            else:
                offspring.add_link(self._crossover_link(dominant_link, recessive_link))

        return offspring

    def _crossover_neuron(self, a: ngen.NeuronGene, b: ngen.NeuronGene):
        neuron_id = a.neuron_id
        bias = self._rng.choose(0.5, a.bias, b.bias)
        activation = self._rng.choose(0.5, a.activation, b.activation)
        return ngen.NeuronGene(neuron_id, bias, activation)

    def _crossover_link(self, a: lgen.LinkGene, b: lgen.LinkGene):
        link_id = a.link_id
        weight = self._rng.choose(0.5, a.weight, b.weight)
        is_enabled = self._rng.choose(0.5, a.is_enabled, b.is_enabled)
        return lgen.LinkGene(link_id, weight, is_enabled)
