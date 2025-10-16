class Neuron:
    b: float
    w: list[float]

    def __init__(self, num_weights: int):
        self.b = 0
        self.w = [0 for _ in range(num_weights)]

    def eval(self, values: list[float]) -> float:
        z = 0
        # compute the weighted sum
        for i in range(len(self.w)):
            z += values[i] * self.w[i]
        # add the bias
        z += self.b
        return z

    def test(self, data: list[list[float]], max_iter: int):
        features = len(data[0]) - 1
        # update the weights based on data
        for _ in range(max_iter):
            # loop through the features in inside dataset
            for d in data:
                # eval result, update w and b if prediction is incorrect
                z = self.eval(d[0:features])
                # label, which is last element in line
                y = d[-1]
                # weighted sum, times binary {-1, 1} label
                if z*y <= 0:
                    # update weights
                    for i in range(len(self.w)):
                        self.w[i] = self.w[i] + y*d[i]
                    # update bias
                    self.b = y
