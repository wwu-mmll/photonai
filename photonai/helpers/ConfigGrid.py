from ..optimization.Hyperparameters import PhotonHyperparam, FloatRange, IntegerRange, BooleanSwitch, Categorical


def create_global_config(pipeline_elements):
    global_hyperparameter_dict = {}
    for p_element in pipeline_elements:
        if len(p_element.hyperparameters) > 0:
            for h_key, h_value in p_element.hyperparameters.items():
                if isinstance(h_value, list):
                    global_hyperparameter_dict[h_key] = h_value
                elif isinstance(h_value, PhotonHyperparam):
                    # when we have a range we need to convert it to a definite list of values
                    if isinstance(h_value, FloatRange) or isinstance(h_value, IntegerRange):
                        # build a definite list of values
                        h_value.transform()
                        global_hyperparameter_dict[h_key] = h_value.values
                    elif isinstance(h_value, BooleanSwitch) or isinstance(h_value, Categorical):
                        global_hyperparameter_dict[h_key] = h_value.values
    return global_hyperparameter_dict
