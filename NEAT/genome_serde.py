import os
import pickle

import NEAT.genome as gen


def serialize_genome(genome: gen.Genome, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(genome, file)


def deserialize_genome(filename: str) -> gen.Genome:
    with open(filename, "rb") as file:
        genome = pickle.load(file)
    return genome
