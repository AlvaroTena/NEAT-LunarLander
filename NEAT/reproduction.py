import copy
import random
from collections import defaultdict
from typing import Dict, List

import NEAT.config as cfg
import NEAT.crossover as cross
import NEAT.indexer as idx
import NEAT.individual as ind
import NEAT.mutator as mut
import NEAT.species as spc
import numpy as np
from RNG.random import Rng


class Reproduction:
    def __init__(
        self,
        genome_config: cfg.GenomeConfig,
        reproduction_config: cfg.ReproductionConfig,
        genome_indexer: idx.Indexer,
        neuron_indexer: idx.Indexer,
        rng: Rng,
    ):
        self._genome_config = genome_config
        self._reproduction_config = reproduction_config
        self._genome_indexer = genome_indexer
        self._neuron_indexer = neuron_indexer
        self._rng = rng
        self._genome_mutator = mut.GenomeMutator(
            genome_config, rng, genome_indexer, neuron_indexer
        )
        self._individual_crossover = cross.IndividualCrossover(genome_indexer, rng)

    def new_population(self, population_size: int):
        individuals: List[ind.Individual] = []
        for _ in range(population_size):
            genome = self._genome_mutator.new_genome()
            individuals.append(ind.Individual(genome))
        return individuals

    # Produces new generation.
    def reproduce(self, population_size: int, species_set: spc.SpeciesSet):
        new_population: List[ind.Individual] = []
        if not species_set.species():
            return new_population

        spawn_sizes = self._compute_spawn_sizes(population_size, species_set)

        for species_id, target_spawn_size in spawn_sizes.items():
            spawn_size = max(target_spawn_size, self._reproduction_config.elitism)
            assert spawn_size > 0

            old_members = self._sorted_old_members(
                species_set.species().get(species_id)
            )

            # Transfer elites
            for i in range(min(self._reproduction_config.elitism, len(old_members))):
                new_population.append(old_members[i])
                spawn_size -= 1

            # Only use the survival threshold fraction to use as parents for the next
            # generation (minimum 2).
            reproduction_cutoff = int(
                min(
                    len(old_members),
                    max(
                        2,
                        np.ceil(
                            self._reproduction_config.survival_threshold
                            * len(old_members)
                        ),
                    ),
                )
            )

            # Randomly choose parents and produce the offsprings allotted to the
            # species.
            for _ in range(spawn_size):
                parent1: ind.Individual = self._rng.choose_random(
                    old_members[:reproduction_cutoff]
                )
                parent2: ind.Individual = self._rng.choose_random(
                    old_members[:reproduction_cutoff]
                )
                offspring = self._individual_crossover.crossover(parent1, parent2)
                self._genome_mutator.mutate(offspring)
                new_population.append(ind.Individual(offspring))

        return new_population

    def _sorted_old_members(self, species: spc.Species):
        old_members = copy.deepcopy(species.members())
        return sorted(
            old_members, key=lambda individual: individual.fitness, reverse=True
        )

    # Returns species ID to spawn amounts.
    def _compute_spawn_sizes(self, population_size: int, species_set: spc.SpeciesSet):
        total_adjusted_fitness = sum(
            species.adjusted_fitness() for species in species_set.species().values()
        )
        spawn_sizes: Dict[int, int] = {}
        total_spawn = 0

        for species_id, species in species_set.species().items():
            spawn_size = self._compute_spawn_size(
                population_size,
                self._reproduction_config.min_species_size,
                species,
                total_adjusted_fitness,
            )
            spawn_sizes[species_id] = spawn_size
            total_spawn += spawn_size

        # Normalize spawn sizes
        normalization_factor = population_size / total_spawn
        for species_id in spawn_sizes:
            spawn_sizes[species_id] = round(
                spawn_sizes[species_id] * normalization_factor
            )

        return spawn_sizes

    def _compute_spawn_size(
        self,
        population_size: int,
        min_species_size: int,
        species: spc.Species,
        total_adjusted_fitness: float,
    ):
        assert species.adjusted_fitness() >= 0.0
        target_size = max(
            min_species_size,
            (species.adjusted_fitness() / total_adjusted_fitness) * population_size,
        )
        # Smooth the change by choosing the mid-point between prev and target
        # size.
        delta = (target_size - len(species.members())) * 0.5
        abs_delta = max(1, abs(round(delta)))
        return (
            len(species.members()) + abs_delta
            if delta > 0
            else len(species.members()) - abs_delta
        )
