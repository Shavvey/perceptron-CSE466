from neuron import Neuron


class NeuralNetwork:
    # NOTE: the more 'correct' name for these would be hidden layers
    layers: list[list[Neuron]]
    # this will store the output of our model
    output: list[float]

    def __init__(self, layers: list[list[Neuron]]):
        """Takes pretrained neurons as input inside `layers`"""
        self.layers = layers 

    def test(self, data: list[list[float]], classes: list[str]) -> list[str]:
        pass

