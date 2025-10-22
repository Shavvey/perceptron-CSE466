from boolean import Boolean


class ConfusionMatrix:
    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float

    def __init__(self, preds: list[Boolean], actuals: list[Boolean]):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i, _ in enumerate(preds):
            is_same = (
                actuals[i].value - preds[i].value
            ) == 0  # check if pred and actual agree
            match is_same:
                case True:
                    if preds[i] == Boolean.YES:
                        true_positive += 1  # true positive
                    else:
                        true_negative += 1  # true negative
                case False:
                    if preds[i] == Boolean.YES:
                        false_positive += 1  # false positive
                    else:
                        false_negative += 1  # false negative
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative


class Stats:
    @staticmethod
    def MSE(preds: list[Boolean], actuals: list[Boolean]) -> float:
        sum = 0
        n = len(preds)
        for i, _ in enumerate(preds):
            sum += (actuals[i].value - preds[i].value) ** 2
        sum /= n
        return sum

    @staticmethod
    def get_cmatrix(preds: list[Boolean], actuals: list[Boolean]) -> ConfusionMatrix:
        return ConfusionMatrix(preds, actuals)
