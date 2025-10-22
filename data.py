import csv as csv
from boolean import Boolean

"""This file is meant to preprocess the iris dataset,
such that we have 7 examples of our target class, along
with 7 non-examples (examples of other classes our
model must reject)"""

FEATURES = [0, 1]  # the selected features of our dataset (in feature indices)
CLASS = "setosa"
DATA_PATH = "iris_rnd_train.csv"
NUM_DATA = 7  # number of examples for class our chosen class, as well as non-examples (not our chosen class)


def get_data(
    data_class: str, num_data: int | None = None, features: list[int] | None = None
) -> list[list[float]]:
    """Pick out features we will use for traning inside iris dataset.
    Optional params dictate how many example and what features to take."""
    if num_data == None:
        num_data = 1 << 32  # get all data
    dataset = []
    class_found = 0
    nclass_found = 0
    with open(DATA_PATH) as csv_file:
        reader = csv.reader(csv_file)
        header = reader.__next__()
        if features == None:
            features = [i for i in range(0, len(header) - 1)]  # select all features
        for line in reader:
            label_idx = len(line) - 1  # label will be in the last idx of line
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


if __name__ == "__main__":
    data = get_data(CLASS)
    for d in data:
        print(d)
