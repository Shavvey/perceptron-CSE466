import csv as csv

"""This file is meant to preprocess the iris dataset,
such that we have 7 examples of our target class, along
with 7 non-examples (examples of other classes our
model must reject)"""

FEATURES = [0, 1] # the selected features of our dataset (in feature indices)
CLASS = "setosa"
DATA_PATH = "iris_rnd_train.csv"
NUM_DATA = 7 # number of examples for class our chosen class, as well as non-examples (not our chosen class)


def get_data() -> list[list[float]]:
    """Pick out features we will use for traning inside iris dataset."""
    dataset = []
    class_found = 0
    nclass_found = 0
    with open(DATA_PATH) as csv_file:
        reader = csv.reader(csv_file)
        _ = reader.__next__()  # throw away header
        for line in reader:
            label_idx = len(line) - 1 # label will be in the last idx of line
            if line[label_idx] == CLASS and class_found < NUM_DATA:
                class_found += 1
                # cast features into floats
                data = [float(line[i]) for i in FEATURES]
                data.append(1.0)  # append binary lable
                dataset.append(data)
            if line[label_idx] != CLASS and nclass_found < NUM_DATA:
                nclass_found += 1
                # cast features into floats
                data = [float(line[i]) for i in FEATURES]
                data.append(-1.0)  # append binary lable
                dataset.append(data)
    return dataset


if __name__ == "__main__":
    data = get_data()
    for d in data:
        print(d)
