class Neuron:
    b: float
    w: list[float]

    def __init__(self, num_weights: int):
        self.b = 0
        # init weights to zero
        self.w = [0 for _ in range(num_weights)]

    def eval(self, inputs: list[float]) -> float:
        """Takes a list of inputs (features in dataset)
        and computes weighted sum of Neuron."""
        z = 0
        # compute the weighted sum
        for i in range(len(self.w)):
            z += inputs[i] * self.w[i]
        # add the bias
        z += self.b
        return z

    def train(self, data: list[list[float]], iters: int):
        """Train the neuron based on data provided"""
        # features inside dataset (last element should we label)
        num_features = len(data[0]) - 1
        # update the weights based on data
        for _ in range(iters):
            # loop through the features in inside dataset
            for d in data:
                # eval result, update w and b if prediction is incorrect
                z = self.eval(d[0:num_features])
                # label, which is last element in line
                y = d[-1]
                # weighted sum, times binary {-1, 1} label
                if z * y <= 0:
                    # update weights
                    for i in range(len(self.w)):
                        self.w[i] = self.w[i] + y * d[i]
                    # update bias
                    self.b = y

    def test(self, data: list[list[float]]) -> list[float]:
        results: list[float] = []
        # extract data point, use feature as params in weighted sum
        for d in data:
            z = self.eval(d)
            # output 1 if weighted sum is greater than bias, 0 otherwise
            res = 1 if z > 0 else 0
            results.append(res)
        return results
