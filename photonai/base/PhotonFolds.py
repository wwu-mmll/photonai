

class FoldInfo:

    def __init__(self, fold_id=None, train_indices: list = None, test_indices: list = None):
        self.fold_id = fold_id
        self.train_indices = train_indices
        self.test_indices = test_indices


class OuterFoldManager:

    def __init__(self, outer_fold_info: FoldInfo, inner_fold_infos: list, get_optimizer_fnc):
        self.outer_fold_info = outer_fold_info
        self.inner_fold_infos = inner_fold_infos
        self.get_optimizer_fnc = get_optimizer_fnc


        self.current_best_config = None

    # How to get optimizer instance?

    def prepare(self, pipeline_elements: list):
        pass

    def fit(self, X, y=None, **kwargs):
        pass


class InnerFoldManager:

    def __init__(self, fold_info: FoldInfo):
        pass
