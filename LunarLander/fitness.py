import concurrent.futures
import os
from typing import List

import gymnasium as gym
import LunarLander.controllers as ctrl
import NEAT.feed_forward_nn as ffnn
import NEAT.individual as ind
import numpy as np
from RNG.random import Rng


def create_environment(render: bool = True):
    if render:
        return gym.make("LunarLander-v2", render_mode="human")
    else:
        return gym.make("LunarLander-v2")


class ComputeFitnessFn:
    def __init__(self, rng: Rng):
        self._rng = rng

    def __call__(
        self, population: List[ind.Individual], generation: int, num_generations: int
    ):
        render_generation = (
            generation % (num_generations * 0.1) == 0
            or generation == num_generations - 1
        )
        selected_individual = self.select_individual(population, generation)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(max(1, os.cpu_count() * 0.7))
        ) as executor:
            futures = {
                executor.submit(
                    self.update_fitness,
                    individual,
                    render_generation and idx == selected_individual,
                ): idx
                for idx, individual in enumerate(population)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    population[idx].fitness = future.result()
                except Exception as exc:
                    print(f"Individual {idx} generated an exception: {exc}")

    def select_individual(
        self, population: List[ind.Individual], generation: int
    ) -> int:
        if generation == 0:
            return self._rng.next_int(len(population))
        else:
            return np.argmax(
                [ind.fitness for ind in population if ind.fitness is not None]
            )

    def update_fitness(
        self, individual: ind.Individual, render_individual=False
    ) -> float:
        max_rounds = 50

        fitness = 0.0
        for round in range(max_rounds):
            render_round = round in {0, max_rounds // 2, max_rounds - 1}
            with create_environment(render=(render_round and render_individual)) as env:
                observation, info = env.reset()
                nn = ffnn.FeedForwardNeuralNetwork.create_from_genome(individual.genome)
                controller = ctrl.AIController(env, nn, self._rng, False, False)

                done = False
                while not done:
                    action = controller.get_action(observation)
                    observation, reward, terminated, truncated, info = env.step(action)

                    if observation[1] == 0 and action == 2:
                        X = -0.3
                    elif observation[1] == 0 and (action == 1 or action == 3):
                        X = -3
                    else:
                        X = 0

                    fitness += reward + X
                    done = terminated or truncated

        return fitness / max_rounds
