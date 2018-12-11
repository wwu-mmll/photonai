from .OptimizationStrategies import PhotonBaseOptimizer
from .fabolas.Fabolas import Fabolas
import datetime


class FabolasOptimizer(PhotonBaseOptimizer):

    def __init__(self, **fabolas_params):
        self._fabolas_params = fabolas_params
        self._fabolas = None
        self.ask = self.next_config_generator()
        self.last_request_time = None
        self.maximize_metric = True

    def prepare(self, pipeline_elements, maximize_metric):
        self.maximize_metric = maximize_metric
        self._fabolas_params.update({'pipeline_elements': pipeline_elements})
        self._fabolas = Fabolas(**self._fabolas_params)
        self.ask = self.next_config_generator()

    def next_config_generator(self):
        self.last_request_time = datetime.datetime.now()
        yield from self._fabolas.calc_config()

    def request_special_params(self):
        return self._fabolas._get_special_params()

    def tell(self, config, performance):
        score = performance[1]
        cost = (datetime.datetime.now() - self.last_request_time).total_seconds()
        self._fabolas.process_result(config, score, cost)
