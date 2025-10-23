from data import get_actuals, get_data
from neuron import Neuron
from stats import ConfusionMatrix, Stats

DATA_CLASS = "versicolor"

def main():
    train_data = get_data(DATA_CLASS, 39)
    # create neuron, makes weights the same as the number of features
    n = Neuron(len(train_data[0]) - 1)
    n.train(train_data, 1000)
    test_data = get_data(DATA_CLASS)
    preds = n.test(test_data)
    actuals = get_actuals(test_data)
    cm = ConfusionMatrix(preds, actuals)
    print(cm)
    pi = Stats.percent_incorrect(preds, actuals)
    print(f"% Incorrect: {pi:.2f}")
    f1 = Stats.f1_score(preds, actuals)
    print(f"F1 Score: {f1}")
    print("Precision: ", cm.precision())
    print("Recall: ", cm.recall())
    print(n)

if __name__ == "__main__":
    main()
