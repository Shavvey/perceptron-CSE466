import csv as csv

"""This file is meant to preprocess the iris dataset,
such that we have 7 examples of our target class, along
with 7 non-examples (examples of other classes our
model must reject)"""

FEATURES = [0, 1]
CLASS = "setosa"
DATA_PATH = "iris_rnd_train.csv"
NUM_DATA = 7


def get_data() -> list[list[float]]:
    """Pick out features we will use for traning inside iris dataset."""
    dataset = []
    class_found = 0
    nclass_found = 0
    with open(DATA_PATH) as csv_file:
        reader = csv.reader(csv_file)
        _ = reader.__next__()  # throw away header
        for line in reader:
            if line[4] == CLASS and class_found < NUM_DATA:
                class_found += 1
                data = [float(line[i]) for i in FEATURES]
                data.append(1.0)  # append binary lable
                dataset.append(data)
            if line[4] != CLASS and nclass_found < NUM_DATA:
                nclass_found += 1
                data = [float(line[i]) for i in FEATURES]
                data.append(-1.0)  # append binary lable
                dataset.append(data)
    return dataset


if __name__ == "__main__":
    data = get_data()
    for d in data:
        print(d)
