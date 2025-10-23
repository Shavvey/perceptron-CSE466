from data import get_actuals, get_data
from neuron import Neuron
from boolean import Boolean
from stats import ConfusionMatrix, Stats


def main():
    train_data = get_data("setosa", 35)
    # create neuron, makes weights the same as the number of features
    n = Neuron(len(train_data[0]) - 1)
    n.train(train_data, 100)
    test_data = get_data("setosa")
    preds = n.test(test_data)
    actuals = get_actuals(test_data)
    cm = ConfusionMatrix(preds, actuals)
    print(cm)
    pi = Stats.percent_incorrect(preds, actuals)
    print(f"% Incorrent: {pi:.2f}")
    f1 = Stats.f1_score(preds, actuals)
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()
