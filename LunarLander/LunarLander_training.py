import argparse
import sys
from typing import List
import numpy as np

import gymnasium as gym

sys.path.append("..")

import LunarLander.fitness as fit
import matplotlib.pyplot as plt
import NEAT.config as cfg
import NEAT.formatter as fmtr
import NEAT.genome_serde as gen_serde
import NEAT.population as pop
import NEAT.reporter as rptr
import NEAT.individual as ind
import networkx as nx
from RNG.random import Rng


def run_training(
    rng: Rng,
    population_size: int,
    num_generations: int,
    best_in_generation_output_dir: str,
    fitness_output_file: str,
    input_pkl_file: str = None
):
    with gym.make("LunarLander-v2") as env:
        obs_action = (env.observation_space.shape[0], env.action_space.n)

    genome_config = cfg.GenomeConfig(
        num_inputs=obs_action[0], num_outputs=obs_action[-1]
    )
    neat_config = cfg.NeatConfig(population_size, False)
    neat_config.genome = genome_config
    compute_fitness = fit.ComputeFitnessFn(rng)

    if input_pkl_file is None:
        population = pop.Population(neat_config, rng)
    else:
        gnome = gen_serde.deserialize_genome(input_pkl_file)
        fitness: float = -np.inf
        individual: ind.Individual = ind.Individual(gnome, fitness)
        individuals: List[ind.Individual] = [individual] * population_size
        population = pop.Population(neat_config, rng, individuals)

    population.register_reporter(rptr.StatsReporter())
    population.register_reporter(
        rptr.BestGenomeSnapshotter(best_in_generation_output_dir)
    )
    population.register_reporter(rptr.FitnessFunctionReporter(fitness_output_file))

    return population.run(compute_fitness, num_generations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--population_size", type=int, help="Size of the population."
    )
    parser.add_argument(
        "-g", "--num_generations", type=int, help="Number of generations."
    )
    parser.add_argument("-o", "--output_file", help="File to save the winning genome.")
    parser.add_argument(
        "-og",
        "--best_in_generation_output_dir",
        help="Directory to save the best genome of each generation.",
    )
    parser.add_argument(
        "-of", "--fitness_output_file", help="File to save fitness scores."
    )
    parser.add_argument(
        "-i", "--input_pkl_file",
        help="pkl file to start new training.",
        default=None
    )
    args = parser.parse_args()

    population_size = args.population_size
    num_generations = args.num_generations
    output_file = args.output_file
    best_in_generation_output_dir = args.best_in_generation_output_dir
    fitness_output_file = args.fitness_output_file
    input_pkl_file = args.input_pkl_file

    rng = Rng(seed_config=42)
    winner = run_training(
        rng,
        population_size,
        num_generations,
        best_in_generation_output_dir,
        fitness_output_file,
        input_pkl_file
    )
    gen_serde.serialize_genome(winner.genome, output_file)
    print(f"Saved winner genome at: {output_file}")
    print(f"Winner fitness: {winner.fitness}")
    # print(fmtr.ForAnimationFormatterFn(winner.genome)())

    G = fmtr.ForAnimationFormatterFn(winner.genome).to_networkx_graph()
    pos = nx.multipartite_layout(G, subset_key="layer")

    node_colors = {0: "#FBE96A", 1: "#C3BD92", 2: "#69E9FA"}
    node_color_map = [
        node_colors[data["layer"]] for node_id, data in G.nodes(data=True)
    ]
    edge_color_map = ["green" if G[u][v]["weight"] > 0 else "red" for u, v in G.edges()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    abs_weights = [abs(w) for w in weights]
    max_weight = max(abs_weights)
    widths = [5 * w / max_weight for w in abs_weights]

    plt.figure(figsize=(15, 10))
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=node_color_map,
        edge_color=edge_color_map,
        width=widths,
    )
    plt.savefig(f"{output_file}_netgraph.png", format="png", dpi=300)
    # plt.show()
