import csv as csv
from math import floor
from boolean import Boolean

"""This file is meant to preprocess the iris dataset,
such that we have `x` examples of our target class, along
with `x` non-examples (examples of other classes our
model must reject)"""

DATA_PATH = "iris_rnd_train.csv"  # hard-coded csv path


def get_data(
    data_class: str, num_data: int | None = None, features: list[int] | None = None
) -> list[list[float]]:
    """Pick out features we will use for traning inside iris dataset.
    Optional params dictate how many examples and what features to take.
    Need to specify class (flower species) to train the neuron/perceptron on."""
    if num_data == None:
        num_data = 1 << 32  # get all data
    dataset = []
    class_found = 0
    nclass_found = 0
    with open(DATA_PATH) as csv_file:
        reader = csv.reader(csv_file)
        header = reader.__next__()
        label_idx = len(header) - 1  # label will be in the last idx of line
        if features == None:
            features = [
                i for i in range(0, len(header) - 1)
            ]  # select all features if none specified in params
        for line in reader:
            if line[label_idx] == data_class and class_found < num_data:
                class_found += 1
                # cast features into floats
                data = [float(line[i]) for i in features]
                data.append(Boolean.YES.value)  # append binary yes label
                dataset.append(data)
            if line[label_idx] != data_class and nclass_found < num_data:
                nclass_found += 1
                # cast features into floats
                data = [float(line[i]) for i in features]
                data.append(Boolean.NO.value)  # append binary no label
                dataset.append(data)
    return dataset


def get_actuals(data: list[list[float]]) -> list[Boolean]:
    """Return all the labels of the data, using our special boolean enum"""
    # assumes that the last part of each row in data is the label
    return [Boolean(d[-1]) for d in data]

def get_train_test_split(data: list[list[float]], train_ratio: float) -> tuple[list[list[float]], list[list[float]]]:
    train_idx = floor(len(data) * train_ratio)
    return (data[:train_idx], data[train_idx:])


def get_actual_classes() -> list[str]:
    preds = []
    with open(DATA_PATH) as csv_file:
        reader = csv.reader(csv_file)
        _ = reader.__next__()  # throw away header
        for line in reader:
            preds.append(line[-1])
    return preds


def get_test_data() -> list[list[float]]:
    with open(DATA_PATH) as csv_file:
        data: list[list[float]] = []
        reader = csv.reader(csv_file)
        _ = reader.__next__()  # throw away header
        for line in reader:
            llen = len(line)
            if line[-1] == '':
                row = [float(data) for data in line[0:llen-1]]
                data.append(row)
    return data


if __name__ == "__main__":
    data = get_data('setosa', 35)
    for i, d in enumerate(data):
        print(f"{i+1}:{d}")
