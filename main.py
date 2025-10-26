from data import get_actuals, get_data
from neuron import Neuron
from stats import ConfusionMatrix, Stats
from boolean import Boolean
from nnetwork import NeuralNetwork

DATA_CLASS = "setosa"


def print_statistics(preds: list[Boolean], actuals: list[Boolean]):
    cm = ConfusionMatrix(preds, actuals)
    print(cm)
    print(f"Incorrect %: {Stats.percent_incorrect(preds, actuals):.2f}")
    print(f"F1 Score: {Stats.f1_score(preds, actuals):.2f}")
    print(f"Precision: {cm.precision():.2f}\nRecall: {cm.recall():.2f}")


def train_two_features():
    FEATURES = [i for i in range(0, 4)]
    best_neuron: Neuron = Neuron(0)  # find best neuron from picked features
    best_features = []  # find best features
    best_f1_score = 0  # use f1 score to find optimal features + neuron config
    for first in FEATURES:  # pick first feature
        for second in [
            feature for feature in FEATURES if feature != first
        ]:  # pick second feature
            feature_pair = [first, second]
            train_data = get_data(DATA_CLASS, 39, feature_pair)
            test_data = get_data(DATA_CLASS, features=feature_pair)
            n = Neuron(len(train_data[0]) - 1)
            n.train(train_data, 100)
            preds = n.test(test_data)
            actuals = get_actuals(test_data)
            f1_score = Stats.f1_score(preds, actuals)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_neuron = n
                best_features = feature_pair
    # get predictions for the best neuron
    test_data = get_data(DATA_CLASS, features=best_features)
    preds = best_neuron.test(test_data)
    actuals = get_actuals(test_data)
    print_statistics(preds, actuals)
    print(f"Best features: {best_features}")
    print(best_neuron)


def train_all_features():
    train_data = get_data(DATA_CLASS, 39)
    test_data = get_data(DATA_CLASS)
    n = Neuron(len(train_data[0]) - 1)
    n.train(train_data, 100)
    preds = n.test(test_data)
    actuals = get_actuals(test_data)
    print_statistics(preds, actuals)


def main():
    network = NeuralNetwork.make_iris_network()
    data = get_data(DATA_CLASS)
    data_classes = ["setosa", "virginica", "versicolor"]
    network.test(data, data_classes)


if __name__ == "__main__":
    main()
