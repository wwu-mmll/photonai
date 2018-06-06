
class MinimumPerformance:

    def __init__(self,  metric, smaller_than):
        self.metric = metric
        self.smaller_than = smaller_than

    def shall_continue(self, inner_folds):
        if inner_folds[0].validation.metrics[self.metric] < self.smaller_than:
            return False
        else:
            return True

