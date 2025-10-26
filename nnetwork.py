from neuron import Neuron
from data import get_data
from stats import Stats
import util


class NeuralNetwork:
    # NOTE: the more 'correct' name for these would be hidden layers
    layers: list[list[Neuron]]
    # this will store the output of our model
    output: list[float]

    def __init__(self, layers: list[list[Neuron]]):
        """Takes pretrained neurons as input inside `layers`"""
        self.layers = layers

    def test(self, data: list[list[float]], data_classes: list[str]):
        outputs = []
        xs = [] # inputs to layers
        ys = [] # output to layers
        for row in data:
            xs = row
            for layer in self.layers:
                for neuron in layer:
                    y = neuron.eval(xs)
                    ys.append(y)
                xs = list.copy(
                    ys
                )  # new inputs will come from prev layer outputs
                ys.clear()
            outputs.append([Stats.sigmoid(x) for x in xs]) # tap xs once loop is done
        # make prediction to terms classses
        pred_classes: list[str] = []
        for output in outputs:
            idx = util.idx_max(output)
            pred = data_classes[idx]
            print(pred)
            pred_classes.append(pred)
        return pred_classes
            

    @staticmethod
    def make_iris_network() -> "NeuralNetwork":
        """Simple function to return network for iris dataset"""
        data_classes = ["setosa", "virginica", "versicolor"]
        # first train the perceptrons
        neurons = []
        for data_class in data_classes:
            train_data = get_data(data_class, 30)
            n = Neuron(len(train_data[0]) - 1)
            n.train(train_data, 100)
            neurons.append(n)
        return NeuralNetwork([neurons])
