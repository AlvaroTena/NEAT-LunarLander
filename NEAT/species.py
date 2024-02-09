from typing import Dict, List

import NEAT.config as cfg
import NEAT.genome as gen
import NEAT.genomic_distance as gen_dis
import NEAT.indexer as idx
import NEAT.individual as ind
import numpy as np


class Species:
    def __init__(self, species_id: int, generation: int, representative: gen.Genome):
        self._species_id = species_id
        self._generation = generation
        self._representative = representative
        self._last_improved = generation
        self._max_fitness = -np.inf
        self._fitness = -np.inf
        self._adjusted_fitness = -np.inf
        self._members: List[ind.Individual] = []
        self._fitness_history: List[float] = []

    def clear_members(self):
        self._members.clear()

    def add_member(self, member: ind.Individual):
        self._members.append(member)

    def representative(self):
        return self._representative

    def update_representative(self, representative: gen.Genome):
        self._representative = representative

    def members(self):
        return self._members

    def id(self):
        return self._species_id

    def generation(self):
        return self._generation

    def last_improved(self):
        return self._last_improved

    def fitness(self):
        return self._fitness

    def adjusted_fitness(self):
        return self._adjusted_fitness

    def size(self):
        return len(self._members)

    def update_fitness(self, generation: int):
        self._fitness = self._compute_fitness()
        if self._fitness > self._max_fitness:
            self._last_improved = generation
            self._max_fitness = self._fitness

    def update_adjusted_fitness(self, adjusted_fitness: float):
        assert adjusted_fitness >= 0.0
        self._adjusted_fitness = adjusted_fitness

    def _compute_fitness(self):
        assert self._members
        return max(
            member.fitness for member in self._members if member.fitness is not None
        )


class SpeciesSet:
    def __init__(self, config: cfg.CompatibilityConfig, species_indexer: idx.Indexer):
        self._config = config
        self._genomic_distance_evaluator = gen_dis.GenomicDistanceEvaluator(config)
        self._species_indexer = species_indexer
        self._species: Dict[int, Species] = {}
        self._fitness = -np.inf

    # Assign each individual to species.
    # The lifetime of a single species spans over multiple generations.
    # However, each individual may be reassigned to different species in every
    # generation. This is due to recomputing species assignments of the whole
    # population in each generation.
    # Each species has a representative genome.
    # Individuals are assigned to the most similar species that satisfies the
    # criteria. The criteria for assigning an individual to species is that the
    # genome distance between that individual and representative is smaller than
    # the configured threshold. Note that the representative used for the
    # species is from the previous generation.
    # # TODO: Is this true?
    # This
    # means that an individual that was the representative of some species may
    # not be part of those species anymore (depending on the iteration order).
    # It's unclear if this is good behaviour or not though.
    # Representatives of each species are updated in each generation.
    # The individual that is the most similar to the current representative
    # becomes the new representative. It's unlikely that the same individual
    # will be representative for multiple species because the genomic distance
    # between different species should be higher than some threshold. However,
    # it's not impossible and in those rare cases the algorithm should still
    # work fine: the side-effect is that there will be two species that are
    # similar to each other and new members will be assigned to the one that
    # comes first in the iteration order. Eventually, one of the species will
    # become small and it will go extinct.
    # New species is created when there's an individual that cannot be assigned
    # to any of the existing species. This individual becomes the
    # representative.
    # Species without any members after speciation are deleted.
    # Implementation notes:
    #   1) Clear all assignments of individuals to species.
    #   2) Update representatives for existing species.
    #      This is done by choosing the individual closest to the current
    #      representative.
    #   3) Assign each individual from the population to any species that
    #   satisfies the criteria.
    #      If no such species, create new species with this individual as
    #      representative.
    # TODO: Original implementation stores genome_to_species, but it's unclear
    # why

    def speciate(self, population: List[ind.Individual], generation: int):
        assert population
        for species in self._species.values():
            # 1) Clear all assignments.
            species.clear_members()

            # 2) Update representatives.
            smallest_distance = np.inf
            new_representative = None
            for individual in population:
                candidate_distance = self._genomic_distance_evaluator.compute(
                    individual.genome, species.representative()
                )
                if smallest_distance >= candidate_distance:
                    smallest_distance = candidate_distance
                    new_representative = individual.genome
            if new_representative is not None:
                species.update_representative(new_representative)

        # 3) Assign each individual to species.
        for individual in population:
            closest = self._find_closest_compatible_species(individual.genome)
            if closest is not None:
                closest.add_member(individual)
            else:
                new_species = Species(
                    self._species_indexer.next(), generation, individual.genome
                )
                new_species.add_member(individual)
                self._species.update({new_species.id(): new_species})

        # TODO: 4) Despite elitism, it can happen that some species doesn't have a member because we clear them here.
        # Maybe this is wrong, maybe I shouldn't clear all the species and should somehow ensure that the fittest
        # remain in their species? But that this hacky. I'd maybe rather remove empty species.
        keys_to_delete = [
            species_id
            for species_id, species in self._species.items()
            if not species.members()
        ]
        for key in keys_to_delete:
            del self._species[key]

        self._genomic_distance_evaluator.clear_cache()

    def remove(self, species_id: int):
        self._species.pop(species_id, None)

    def species(self):
        return self._species

    def empty(self):
        return len(self._species) == 0

    def update_fitness(self, generation: int):
        self._fitness = -np.inf
        for species in self._species.values():
            species.update_fitness(generation)
            self._fitness = max(self._fitness, species.fitness())

        self._update_adjusted_fitness()

    def fitness(self):
        return self._fitness

    def genomic_distance_evaluator(self):
        return self._genomic_distance_evaluator

    def _find_closest_compatible_species(self, candidate: gen.Genome):
        best_distance = np.inf
        closest_species = None

        for species in self._species.values():
            candidate_distance = self._genomic_distance_evaluator.compute(
                species.representative(), candidate
            )
            if (
                candidate_distance < best_distance
                and candidate_distance <= self._config.threshold
            ):
                best_distance = candidate_distance
                closest_species = species

        return closest_species

    def _update_adjusted_fitness(self):
        min_fitness = min(
            member.fitness
            for species in self._species.values()
            for member in species.members()
            if member.fitness is not None
        )
        max_fitness = max(
            member.fitness
            for species in self._species.values()
            for member in species.members()
            if member.fitness is not None
        )
        fitness_range = max(1.0, max_fitness - min_fitness)

        for species in self._species.values():
            mean_species_fitness = self._mean(species.members())
            species.update_adjusted_fitness(
                (mean_species_fitness - min_fitness) / fitness_range
            )

    def _mean(self, members: List[ind.Individual]):
        m = sum(member.fitness for member in members if member.fitness is not None)
        return m / len(members)
