from NEAT.activations import Activation


class NeuronGene:
    def __init__(self, neuron_id: int, bias: float, activation: Activation):
        self.neuron_id = neuron_id
        self.bias = bias
        self.activation = activation
