from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.base.photon_elements import Switch

from photonai.optimization import GridSearchOptimizer, RandomGridSearchOptimizer, \
    SkOptOptimizer, RandomSearchOptimizer


class MetaHPOptimizer(PhotonSlaveOptimizer):
    """
     Searches for the best configuration by randomly testing k possible hyperparameter combinations without grid.
    """

    def __init__(self, **kwargs):

        # get name of optimizer from params
        if not "name" in kwargs:
            raise ValueError("MetaHPOptimizer: Please specify name of optimizer class, "
                             "e.g. random_grid_search with key 'name'")
        self.optimizer_name = kwargs['name']
        if self.optimizer_name == 'smac':
            raise ValueError("MetaHPOptimizer: Using smac does "
                             "not make sense here, as it is by default enabled to handle a switch")
        del kwargs['name']
        self.optimizer_params = kwargs

        self.pipeline_elements = None
        self.current_optimizer = None
        self.n_configurations = -1
        self.estimator_dict = {}
        self.switch_name = ''
        self.switch_estimator_config_key = ''
        self.ask = self.next_config_generator()

        self.OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                                     'random_grid_search': RandomGridSearchOptimizer,
                                     'sk_opt': SkOptOptimizer,
                                     'random_search': RandomSearchOptimizer}

    def prepare(self, pipeline_elements, maximize_metric):
        if not pipeline_elements:
            raise ValueError("Optimizer cannot optimize empty pipeline")

        if not isinstance((pipeline_elements[-1]), Switch):
            raise ValueError("Cannot use switch_optimizer without Switch Element at the end of pipeline.")

        self.pipeline_elements = pipeline_elements
        switch = self.pipeline_elements[-1]
        self.switch_name = switch.name
        self.switch_estimator_config_key = switch.name + "__estimator_name"
        if self.optimizer_name not in self.OPTIMIZER_DICTIONARY:
            raise ValueError("SwitchOptimizer could not find optimizer {}".format(self.optimizer_name))
        optimizer_class = self.OPTIMIZER_DICTIONARY[self.optimizer_name]

        for element in switch.elements:
            optimizer = optimizer_class(**self.optimizer_params)
            # sub_pipeline = pipeline without estimator plus current estimator
            sub_pipeline = [e for e in self.pipeline_elements[:-1]]
            sub_pipeline.append(element)
            optimizer.prepare(sub_pipeline, maximize_metric)
            self.estimator_dict[element.name] = optimizer
        self.ask = self.next_config_generator()

        if "n_configurations" in self.optimizer_params:
            self.n_configurations = len(switch.elements) * self.optimizer_params["n_configurations"]

    def next_config_generator(self):
        for element_name, optimizer in self.estimator_dict.items():
            self.current_optimizer = optimizer
            for config in self.current_optimizer.ask:
                # update key_names
                config_copy = dict()
                for c_key, c_value in config.items():
                    if c_key.startswith(element_name):
                        config_copy[self.switch_name + "__" + c_key] = c_value
                    else:
                        config_copy[c_key] = c_value
                # append the name of the current estimator to config so we can use it to filter the result by estimator
                config_copy[self.switch_estimator_config_key] = element_name
                yield config_copy

    def tell(self, config, performance):
        # influence return value of next_config
        # remove current estimator name as it is just a hack to filter configs afterwards, and not part of the HP space
        config = dict(config)
        del config[self.switch_estimator_config_key]
        config_copy = dict()
        for c_key, c_value in config.items():
            if c_key.startswith(self.switch_name):
                new_key = c_key.replace(self.switch_name + "__", '')
                config_copy[new_key] = c_value
            else:
                config_copy[c_key] = c_value
        self.current_optimizer.tell(config_copy, performance)
