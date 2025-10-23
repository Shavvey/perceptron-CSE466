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

    def precision(self) -> float:
        """Precision is defined as the number of true positives divided by total positives"""
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self) -> float:
        """Recall is defined as the number of true positives divided by the number of actual positives (TP + FN)"""
        return self.true_positive / (self.true_positive + self.false_negative)

    def __str__(self) -> str:
        """Get nice looking string representation of confusion/classifcation matrix"""
        sb: str = f"[ TP: {self.true_positive:02d} | FN: {self.false_negative:02d} ]\n"
        sb += f"[ FP: {self.false_positive:02d} | TN: {self.true_negative:02d} ]"
        return sb


class Stats:
    @staticmethod
    def percent_incorrect(preds: list[Boolean], actuals: list[Boolean]) -> float:
        num_incorrect = 0
        for i, _ in enumerate(preds):
            if preds[i] != actuals[i]:
                num_incorrect += 1
        return (num_incorrect / len(preds)) * 100

    @staticmethod
    def f1_score(preds: list[Boolean], actuals: list[Boolean]) -> float:
        """Defined as the harmonic mean of precision and recall"""
        cm = ConfusionMatrix(preds, actuals)
        # get precision and recall from confusion matrix
        recall = cm.recall()
        precision = cm.precision()
        f1 = (2 * recall * precision) / (precision + recall)
        return f1
