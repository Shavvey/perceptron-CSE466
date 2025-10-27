from neuron import Neuron
from data import get_data, get_actuals
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

    def test(
        self,
        data: list[list[float]],
        data_classes: list[str],
        input_features: list[list[int]] | None = None,
    ):
        """Test perceptron network, input_features optionally let's us discriminate on what input to use.
        This is used when we create a percpetron out of only the two best features for each neuron classifier."""
        outputs = []
        xs = []  # inputs to layers
        ys = []  # output to layers
        for row in data:
            xs = row
            for idx, layer in enumerate(self.layers):
                for i, neuron in enumerate(layer):
                    y = None
                    if input_features == None or idx != 0:
                        y = neuron.eval(xs)
                    elif idx == 0:
                        # only use the features
                        y = neuron.eval([xs[f] for f in input_features[i]])
                    ys.append(y)
                xs = list.copy(ys)  # new inputs will come from prev layer outputs
                ys.clear()
            outputs.append([Stats.sigmoid(x) for x in xs])  # tap xs once loop is done
        # make prediction to terms classses
        pred_classes: list[str] = []
        for output in outputs:
            idx = util.idx_max(output)
            pred = data_classes[idx]
            pred_classes.append(pred)
        return pred_classes

    @staticmethod
    def make_iris_all_feature_network() -> "NeuralNetwork":
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

    @staticmethod
    def make_iris_best_two_feature_network() -> tuple["NeuralNetwork", list[list[int]]]:
        data_classes = ["setosa", "virginica", "versicolor"]
        neurons: list[Neuron] = []
        input_features: list[list[int]] = []
        for data_class in data_classes:
            FEATURES = [i for i in range(0, 4)]
            best_neuron: Neuron = Neuron(0)  # find best neuron from picked features
            best_features = []  # find best features
            best_f1_score = 0  # use f1 score to find optimal features + neuron config
            for first in FEATURES:  # pick first feature
                for second in [
                    feature for feature in FEATURES if feature != first
                ]:  # pick second feature
                    feature_pair = [first, second]
                    train_data = get_data(data_class, 39, feature_pair)
                    test_data = get_data(data_class, features=feature_pair)
                    n = Neuron(len(train_data[0]) - 1)
                    n.train(train_data, 100)
                    preds = n.test(test_data)
                    actuals = get_actuals(test_data)
                    f1_score = Stats.f1_score(preds, actuals)
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_neuron = n
                        best_features = feature_pair
            neurons.append(best_neuron) # get best neuron
            input_features.append(best_features) # get best feaures of neuron
        return (NeuralNetwork([neurons]), input_features)
                
