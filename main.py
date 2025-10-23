from data import get_actuals, get_data
from neuron import Neuron
from boolean import Boolean
from stats import ConfusionMatrix, Stats


def main():
    # get setosa data (35 examples and 35 non-examples)
    data = get_data("setosa", 35)
    # create neuron, makes weights the same as the number of features
    n = Neuron(len(data[0]) - 1)
    n.train(data, 100)
    preds = n.test(data)
    actuals = get_actuals(data)
    cm = ConfusionMatrix(preds, actuals)
    print(cm)
    pi = Stats.percent_incorrect(preds, actuals)
    print(f"% Incorrent: {pi:.2f}")

if __name__ == "__main__":
    main()
