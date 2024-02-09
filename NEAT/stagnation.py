from typing import List

import NEAT.config as cfg
import NEAT.species as spc


class SpeciesData:
    def __init__(self, species_id: int, stagnant_time: int, fitness: float):
        self.species_id = species_id
        self.stagnant_time = stagnant_time
        self.fitness = fitness

    def __lt__(self, other):
        if not isinstance(other, SpeciesData):
            raise NotImplementedError
        return self.fitness < other.fitness


class Stagnation:
    def __init__(self, config: cfg.StagnationConfig):
        self._config = config

    def remove_stagnant(self, species_set: spc.SpeciesSet, generation: int):
        species_data: List[SpeciesData] = []
        for species_id, species in species_set.species().items():
            stagnant_time = generation - species.last_improved()
            species_data.append(
                SpeciesData(species_id, stagnant_time, species.fitness())
            )

        species_data.sort()

        for i, data in enumerate(species_data):
            if self.should_remove(i, data, len(species_data)):
                species_set.remove(data.species_id)

    def should_remove(self, idx: int, data: SpeciesData, num_species: int):
        if num_species - idx <= self._config.elitism:
            # Keep elites
            return False
        return data.stagnant_time >= self._config.max_stagnation
