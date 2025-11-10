from data import get_actuals, get_data, get_test_data, get_train_test_split
from neuron import Neuron
from stats import ConfusionMatrix, Stats
from boolean import Boolean
from nnetwork import NeuralNetwork
from matplotlib import pyplot as pp

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


def graph_train_all_features(flower_species_label: str, max_itrs: int):
    percent_incorrect_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    # train training data
    train_data = get_data(flower_species_label, 39)
    test_data = get_data(flower_species_label)
    # get test data
    # create perceptron
    nn = Neuron(len(train_data[0]) - 1)
    for _ in range(max_itrs):
        preds = nn.test(test_data)
        actuals = get_actuals(test_data)
        pi = Stats.percent_incorrect(preds, actuals)
        f1 = Stats.f1_score(preds, actuals)
        recall = Stats.recall(preds, actuals)
        precision = Stats.precision(preds, actuals)
        # collect
        percent_incorrect_scores.append(pi)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        # make a single train step
        nn.train_step(train_data)
    fig, axs = pp.subplots(2, 2)
    fig.suptitle(f"Performance Graph for {flower_species_label.capitalize()} Training")
    # create graph
    itrs = [itr for itr in range(max_itrs)]
    axs[0, 0].set_title("Incorrect (%) vs. Train Iteration")
    axs[0, 0].plot(itrs, percent_incorrect_scores, "g--")

    axs[0, 1].set_title("F1 Scores vs. Train Iteration")
    axs[0, 1].plot(itrs, f1_scores, "r--")

    axs[1, 0].set_title("Recall Scores vs. Train Iteration")
    axs[1, 0].plot(itrs, recall_scores, "b--")

    axs[1, 1].set_title("Precision Scores vs. Train Iteration")
    axs[1, 1].plot(itrs, precision_scores, "y--")
    pp.show()


def graph_train_two_features(
    flower_species_label: str, max_itrs: int, features: list[int]
):
    percent_incorrect_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    # train training data
    train_data = get_data(flower_species_label, 25, features)
    test_data = get_data(flower_species_label, features=features)
    # get test data
    # create perceptron
    nn = Neuron(len(train_data[0]) - 1)
    for _ in range(max_itrs):
        preds = nn.test(test_data)
        actuals = get_actuals(test_data)
        pi = Stats.percent_incorrect(preds, actuals)
        f1 = Stats.f1_score(preds, actuals)
        recall = Stats.recall(preds, actuals)
        precision = Stats.precision(preds, actuals)
        # collect
        percent_incorrect_scores.append(pi)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        # make a single train step
        nn.train_step(train_data)
    fig, axs = pp.subplots(2, 2)
    fig.suptitle(f"Performance Graph for {flower_species_label.capitalize()} Training")
    # create graph
    itrs = [itr for itr in range(max_itrs)]
    axs[0, 0].set_title("Incorrect (%) vs. Train Iteration")
    axs[0, 0].plot(itrs, percent_incorrect_scores, "g--")

    axs[0, 1].set_title("F1 Scores vs. Train Iteration")
    axs[0, 1].plot(itrs, f1_scores, "r--")

    axs[1, 0].set_title("Recall Scores vs. Train Iteration")
    axs[1, 0].plot(itrs, recall_scores, "b--")

    axs[1, 1].set_title("Precision Scores vs. Train Iteration")
    axs[1, 1].plot(itrs, precision_scores, "y--")
    pp.show()


def get_four_features_preds():
    network = NeuralNetwork.make_iris_all_feature_network()
    data = get_test_data()
    data_classes = ["setosa", "virginica", "versicolor"]
    preds = network.test(data, data_classes)
    for i, pred in enumerate(preds):
        print(f"{i+1}:{pred}")


def get_best_two_features_preds():
    network, input_features = NeuralNetwork.make_iris_best_two_feature_network()
    data = get_test_data()
    data_classes = ["setosa", "virginica", "versicolor"]
    preds = network.test(data, data_classes, input_features)
    for i, pred in enumerate(preds):
        print(f"{i+1}:{pred}")


def print_stats_for_single_classifer(data_class: str, num_examples: int):
    train_data = get_data(data_class, num_examples)
    test_data = get_data(data_class)
    n = Neuron(len(train_data[0]) - 1)
    n.train(train_data, 100)
    print_statistics(n.test(test_data), get_actuals(test_data))


def main():
    # example of how to get combined predictions for best features
    graph_train_two_features("virginica", 30, [0, 2])


if __name__ == "__main__":
    main()
