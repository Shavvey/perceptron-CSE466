from data import get_data
from neuron import Neuron


def main():
    # get setosa data (35 examples and 35 non-examples)
    data = get_data("setosa", 35)
    # create neuron, makes weights the same as the number of features
    n = Neuron(len(data[0]) - 1)
    n.train(data, 100)

if __name__ == "__main__":
    main()
