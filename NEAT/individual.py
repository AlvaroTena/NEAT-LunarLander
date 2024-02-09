import NEAT.genome as gen


class Individual:
    def __init__(self, genome: gen.Genome, fitness: float = None):
        self.genome = genome
        self.fitness = fitness
