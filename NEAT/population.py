from typing import List, Optional

import LunarLander.fitness as fit
import NEAT.activations as act
import NEAT.config as cfg
import NEAT.genome as gen
import NEAT.indexer as idx
import NEAT.individual as ind
import NEAT.reporter as rptr
import NEAT.reproduction as repro
import NEAT.species as spc
import NEAT.stagnation as stgn
from RNG.random import Rng


class Population:

    def __init__(
        self,
        config: cfg.NeatConfig,
        rng: Rng,
        individuals: List[ind.Individual] = None,
        species_set: spc.SpeciesSet = None,
        generation=0,
    ):
        self._config = config
        self._rng = rng
        self._generation = generation
        self._individuals = individuals if individuals is not None else []

        if species_set is not None:
            self._species_set = species_set
            self._species_indexer = idx.Indexer(
                self.max_species_id(self._species_set) + 1
            )
        else:
            self._species_indexer = idx.Indexer(0)
            self._species_set = spc.SpeciesSet(
                self._config.compatibility, self._species_indexer
            )

        if individuals:
            self._genome_indexer = idx.Indexer(
                self.max_genome_id(self._individuals) + 1
            )
            self._neuron_indexer = idx.Indexer(
                self.max_neuron_id(self._individuals) + 1
            )
            self._reproduction = repro.Reproduction(
                self._config.genome,
                self._config.reproduction,
                self._genome_indexer,
                self._neuron_indexer,
                rng,
            )
        else:
            self._genome_indexer = idx.Indexer(0)
            self._neuron_indexer = idx.Indexer(config.genome.num_outputs)
            self._reproduction = repro.Reproduction(
                self._config.genome,
                self._config.reproduction,
                self._genome_indexer,
                self._neuron_indexer,
                rng,
            )
            self._individuals = self._reproduction.new_population(
                config.population_size
            )
            self._species_set.speciate(self._individuals, self._generation)

        self._stagnation = stgn.Stagnation(self._config.stagnation)
        self._reporters = rptr.Reporters()
        self._best_ever = None

    # Run the NEAT algorithm for up to [num_generations].
    #
    # FitnessFn: takes the iterator over [Individual&]
    # and must set the fitness for each individual.
    def run(self, compute_fitness: fit.ComputeFitnessFn, num_generations: int):
        for generation in range(num_generations):
            self._reporters.start_generation(self._generation)

            compute_fitness(self._individuals, generation, num_generations)
            best_in_generation = self._compute_current_best()
            self._update_best_ever(best_in_generation)
            self._reporters.post_evaluate(self._individuals, best_in_generation)
            self._species_set.update_fitness(self._generation)

            # TODO: Add fitness-based termination.

            self._stagnation.remove_stagnant(self._species_set, self._generation)
            self._individuals = self._reproduction.reproduce(
                self._config.population_size, self._species_set
            )

            if not self._species_set.species():
                if self._config.reset_on_extinction:
                    self._individuals = self._reproduction.new_population(
                        self._config.population_size
                    )
                else:
                    raise RuntimeError("Population is completely extinct.")

            self._species_set.speciate(self._individuals, self._generation)
            self._reporters.end_generation(
                self._generation, self._individuals, self._species_set
            )

            self._generation += 1

        self._reporters.end_training()
        return self._best_ever

    def best_ever(self):
        return self._best_ever.genome if self._best_ever else None

    def register_reporter(self, reporter: rptr.Reporter):
        self._reporters.register_reporter(reporter)

    def _compute_current_best(self):
        assert self._individuals
        best = max(self._individuals, key=lambda ind: ind.fitness)
        return best

    def _update_best_ever(self, best_in_generation: ind.Individual):
        if not self._best_ever or self._best_ever.fitness < best_in_generation.fitness:
            self._best_ever = best_in_generation
            print(f"Best ever fitness: {best_in_generation.fitness}")

    @staticmethod
    def max_genome_id(individuals: List[ind.Individual]):
        return max((individual.genome.id() for individual in individuals), default=0)

    @staticmethod
    def max_neuron_id(individuals: List[ind.Individual]):
        return max(
            (
                neuron.neuron_id
                for individual in individuals
                for neuron in individual.genome.neurons()
            ),
            default=0,
        )

    @staticmethod
    def max_species_id(species_set: spc.SpeciesSet):
        return max(species_set.species().keys(), default=0)
