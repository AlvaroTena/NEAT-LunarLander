import gymnasium as gym
import gymnasium.utils.play
import NEAT.feed_forward_nn as ffnn
import NEAT.genome_serde as gen_serde
import numpy as np
import pygame
from RNG.random import Rng


class Controller:
    def get_action(self, observation):
        raise NotImplementedError("This method should be overridden.")


class KeyboardController(Controller):
    def __init__(self, env):
        self.env = env
        self.lunar_lander_keys = {
            (pygame.K_UP,): 2,
            (pygame.K_LEFT,): 1,
            (pygame.K_RIGHT,): 3,
        }

    def play(self):
        gymnasium.utils.play.play(
            self.env, zoom=3, keys_to_action=self.lunar_lander_keys, noop=0
        )


class AIController(Controller):
    def __init__(
        self,
        env: gym.Env,
        nn: ffnn.FeedForwardNeuralNetwork,
        rng: Rng,
        print_inputs=False,
        print_actions=False,
    ):
        self._env = env
        self._nn = nn
        self._rng = rng
        self._print_inputs = print_inputs
        self._print_actions = print_actions
        self._observations = []

    @staticmethod
    def load_model(filename):
        genome = gen_serde.deserialize_genome(filename)
        nn = ffnn.FeedForwardNeuralNetwork.create_from_genome(genome)
        print("Model loaded")
        return AIController(nn)

    def environment(self):
        return self._env

    def last_inputs(self):
        return self._observations

    def get_action(self, observation):
        state = observation

        if self._print_inputs:
            print(f"Inputs: {state}")

        action_probs = self._nn.activate(state)

        prob_sum = sum(action_probs)
        if prob_sum == 0:
            action_probs = np.array(action_probs) + (1.0 / len(action_probs))
        else:
            action_probs = np.array(action_probs) / prob_sum

        if self._print_actions:
            print(f"Outputs: {action_probs}")

        action = self._rng.choose_probability(
            range(len(action_probs)), probabilities=action_probs
        )

        return action
