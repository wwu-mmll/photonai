from itertools import product

from photonai.optimization import PhotonHyperparam, IntegerRange, FloatRange, Categorical, BooleanSwitch
from photonai.photonlogger import logger


def create_global_config_dict(pipeline_elements: list) -> dict:
    """
    Creation of a definition set for grid-based optimizers in format: dict key -> value.

    A grid is generated from a given list of hyperparameters for the optimization process.
    Furthermore, the initialization of hyperparameters takes place.

    Parameters:
        pipeline_elements:
            List of all set hyperparameters.

    Returns:
        Grid of configurations.

    """
    global_hyperparameter_dict = {}
    for p_element in pipeline_elements:

        # global_hyperparameter_dict[p_element.name] = {}

        if len(p_element.hyperparameters) > 0:
            for h_key, h_value in p_element.hyperparameters.items():
                if isinstance(h_value, list):
                    # global_hyperparameter_dict[p_element.name][h_key] = h_value
                    global_hyperparameter_dict[h_key] = h_value
                elif isinstance(h_value, PhotonHyperparam):
                    # when we have a range we need to convert it to a definite list of values
                    if isinstance(h_value, FloatRange) or isinstance(h_value, IntegerRange):
                        # build a definite list of values
                        h_value.transform()
                        # global_hyperparameter_dict[p_element.name][h_key] = h_value.values
                        global_hyperparameter_dict[h_key] = h_value.values
                    elif isinstance(h_value, BooleanSwitch) or isinstance(h_value, Categorical):
                        global_hyperparameter_dict[h_key] = h_value.values
    return global_hyperparameter_dict


def create_global_config_grid(pipeline_elements: list, add_name: str = '') -> list:
    """
    Creation of a list of configuration for grid-based optimizers.
    A grid is generated from a given list of hyperparameters for the optimization process.

    Parameters:
        pipeline_elements:
            List of PipelineElements.

        add_name: str, default=''
            Set prefix to dict keys.

    Returns:
        List of dicts. Every dict is a possible configurations.

    """
    global_hyperparameter_list = []
    for element in pipeline_elements:
        if hasattr(element, "generate_config_grid"):
            config_grid = element.generate_config_grid()
            if len(config_grid) > 0:
                global_hyperparameter_list.append(config_grid)

    praefix = ''
    if add_name != '':
        praefix = add_name + '__'

    # if len(global_hyperparameter_list) == 1:
    #     return [dict((praefix + pair[0], pair[1]) for d in global_hyperparameter_list[0] for pair in d.items())]
    # else:
    threshold = 1000000
    total_product_num = 1
    for i in global_hyperparameter_list:
        total_product_num = total_product_num * len(i)
    if total_product_num > threshold:
        msg = 'The entire configuration grid entails more than ' + str(threshold) + ' possible configurations. ' \
                                                                                    'This might take very ' \
                                                                                    'long to both compute ' \
                                                                                    'and process.'
        logger.error(msg)
        raise ValueError(msg)
    config_list = list(product(*global_hyperparameter_list))
    config_dicts = []
    # get all configs in one
    for c in config_list:
        config_dicts.append(dict((praefix + pair[0], pair[1]) for d in c for pair in d.items()))
    return config_dicts
