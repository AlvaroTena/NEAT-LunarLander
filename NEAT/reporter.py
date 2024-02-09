import os
import time
from typing import List, Tuple

import NEAT.formatter as fmtr
import NEAT.genome_serde as gen_serde
import NEAT.genomic_distance as gen_dis
import NEAT.individual as ind
import NEAT.species as spc
import numpy as np


class Reporter:
    def start_generation(self, generation: int):
        pass

    def end_generation(
        self,
        generation: int,
        population: List[ind.Individual],
        species_set: spc.SpeciesSet,
    ):
        pass

    def post_evaluate(
        self, population: List[ind.Individual], best_in_generation: ind.Individual
    ):
        pass

    def end_training(self):
        pass


class StatsReporter(Reporter):
    def __init__(self, print_genomic_distance=False):
        self._print_genomic_distance = print_genomic_distance
        self._generation_start_time = None

    def start_generation(self, generation):
        print(f"====== Running generation {generation} ======")
        self._generation_start_time = time.time()

    def end_generation(self, generation, population, species_set):
        print(
            f"Population of {len(population)} members in {len(species_set.species())} species."
        )
        if self._print_genomic_distance:
            print(
                "   ID   age   size  fitness  adj fit  stag  avg(gen dist)  stdev(gen dist)"
            )
            print(
                "  ====  ====  ====  =======  =======  ====  =============  ==============="
            )
        else:
            print("   ID   age   size  fitness  adj fit  stag")
            print("  ====  ====  ====  =======  =======  ====")

        keys = sorted(species_set.species().keys())

        for species_id in keys:
            species = species_set.species().get(species_id)
            age = generation - species.generation()
            size = len(species.members())
            fitness = species.fitness()
            adjusted_fitness = species.adjusted_fitness()
            stagnation_time = generation - species.last_improved()

            if self._print_genomic_distance:
                mean_genetic_distance, stdev_genetic_distance = (
                    self._calculate_genomic_distances(
                        species_set.genomic_distance_evaluator(), species
                    )
                )
                print(
                    "  {:>4}  {:>3}  {:>4}  {:>7.2f}  {:>7.5f}  {:>4}  {:>13.2f}  {:>15.2f}".format(
                        species.id(),
                        age,
                        size,
                        fitness,
                        adjusted_fitness,
                        stagnation_time,
                        mean_genetic_distance,
                        stdev_genetic_distance,
                    )
                )
            else:
                print(
                    "  {:>4}  {:>3}  {:>4}  {:>7.2f}  {:>7.5f}  {:>4}".format(
                        species.id(),
                        age,
                        size,
                        fitness,
                        adjusted_fitness,
                        stagnation_time,
                    )
                )

        generation_end_time = time.time()
        seconds = generation_end_time - self._generation_start_time

        evaluator = species_set.genomic_distance_evaluator()
        if self._print_genomic_distance:
            print()
            print("======== Genomic Distance Cache Stats ========")
            print("  size    hits  misses")
            print("  size    hits  misses")
            print("  ======  ====  ======")
            print(
                "  {:>6}  {:>4}  {:>6}".format(
                    evaluator.cache_size(), evaluator.hits(), evaluator.misses()
                )
            )
        print()
        print(f"Generation time: {seconds:.3f} seconds.")

    def _calculate_genomic_distances(
        self,
        genomic_distance_evaluator: gen_dis.GenomicDistanceEvaluator,
        species: spc.Species,
    ) -> Tuple[float, float]:
        members = species.members()
        mean = 0.0
        for individual in members:
            genomic_distance = genomic_distance_evaluator.compute(
                individual.genome, species.representative()
            )
            mean += genomic_distance
            if genomic_distance > 4.0:
                print(
                    "Got genomic distance that's high enough for new species**************************"
                )
        mean /= len(members)

        variance = 0.0
        for individual in members:
            genomic_distance = genomic_distance_evaluator.compute(
                individual.genome, species.representative()
            )
            diff = genomic_distance - mean
            variance = diff * diff
        variance /= len(members)

        if len(species.members()) <= 1:
            return mean, 0.0

        return mean, np.sqrt(variance)


class BestGenomeSnapshotter(Reporter):
    def __init__(self, output_dir: str, print_dot_files=True):
        self._generation = 0
        self._output_dir = output_dir
        self._print_dot_files = print_dot_files
        os.makedirs(output_dir, exist_ok=True)

    def start_generation(self, generation: int):
        self._generation = generation

    def post_evaluate(
        self, population: List[ind.Individual], best_in_generation: ind.Individual
    ):
        pkl_filename = f"winner-generation-{self._generation}.pkl"
        pkl_filepath = os.path.join(self._output_dir, pkl_filename)
        gen_serde.serialize_genome(best_in_generation.genome, str(pkl_filepath))

        # Guardar la representación DOT del mejor genoma
        dot_filename = f"winner-generation-{self._generation}.dot"
        dot_filepath = os.path.join(self._output_dir, dot_filename)
        dot_representation = fmtr.DotFormatterFn(best_in_generation.genome)()

        try:
            with open(dot_filepath, "w") as dot_file:
                dot_file.write(dot_representation)
        except IOError as e:
            print(f"Failed to open or write to the file: {dot_filepath}\nError: {e}")


class FitnessFunctionReporter(Reporter):
    def __init__(self, output: str):
        self._output = output
        self._generation = 0
        self._samples: List[Tuple[int, float]] = []

    def start_generation(self, generation: int):
        self._generation = generation

    def post_evaluate(
        self, population: List[ind.Individual], best_in_generation: ind.Individual
    ):
        self._samples.append((self._generation, best_in_generation.fitness))

    def end_training(self):
        with open(self._output, "w") as f:
            for generation, fitness in self._samples:
                f.write(f"{generation} {fitness}\n")


class Reporters:
    def __init__(self):
        self._reporters: List[Reporter] = []

    def register_reporter(self, reporter: Reporter):
        self._reporters.append(reporter)

    def start_generation(self, generation: int):
        for reporter in self._reporters:
            reporter.start_generation(generation)

    def end_generation(
        self,
        generation: int,
        population: List[ind.Individual],
        species_set: spc.SpeciesSet,
    ):
        for reporter in self._reporters:
            reporter.end_generation(generation, population, species_set)

    def post_evaluate(
        self, population: List[ind.Individual], best_in_generation: ind.Individual
    ):
        for reporter in self._reporters:
            reporter.post_evaluate(population, best_in_generation)

    def end_training(self):
        for reporter in self._reporters:
            reporter.end_training()
