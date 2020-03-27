import json
import sys
import inspect
import numpy as np

from sklearn.model_selection import *
from photonai.optimization.hyperparameters import *
from photonai.base.photon_elements import *

try:
    from photonai.base.hyperpipe import Hyperpipe, OutputSettings
except:
    pass


class JsonTransformer(object):

    def __init__(self, black_list: list = None):
        if black_list is None:
            self.black_list = ["base_element"]
        else:
            self.black_list = black_list
        self.json = {}
        self.attribute_allocator = {"PipelineElement": ["initial_name", "initial_hyperparameters", "test_disabled", "kwargs"],
                                    "Branch": ["initial_name", "elements"],
                                    "Stack": ["initial_name", "elements"],
                                    "Switch": ["initial_name", "elements"],
                                    "NeuroBranch": ["initial_name", "elements"]
                                    }
        # "FloatRange": ["start", "stop", "range_type", "step", "num"]
    @staticmethod
    def write_json_file(value: dict, path: str):
        """
        static method for dumping dict json
        :param json: dict to dump
        :param path: storage path
        :return: None
        """
        with open(path, 'w') as outfile:
            json.dump(value, outfile, indent=4)
        return

    @staticmethod
    def read_json_file(filepath: str):
        """
        reverse function to write_json_file, read json file -> dict
        :param filepath: storage path
        :return: dict type
        """
        with open(filepath, 'r') as outfile:
            value = json.load(outfile)
        return value

    @staticmethod
    def str_to_class(classname: str):
        """
        function for bringing classname to life
        :param classname: String-name of class
        :return: class
        """

        if classname in [x[0] for x in inspect.getmembers(sys.modules[__name__])]:
            obj = getattr(sys.modules[__name__], classname)
        elif classname in [x[0] for x in inspect.getmembers(sys.modules["photonai.base"])]:
            obj = getattr(sys.modules["photonai.base"], classname)
        else:
            msg = "Json Transformer is not able to initialize the hyperpipe. Class: "+ classname + " is not defined."
            logger.error(msg)
            raise ValueError(msg)
        return obj

    def to_json_file(self, pipe, path: str):
        """
        main function for saving PHOTON.Hyperpipe -> file.json
        :param pipe:
        :return:
        """
        self.json = self.create_json(pipe)
        self.write_json_file(self.json, path)

    def create_json(self, pipe):
        """
        function for saving a PHOTON.Hyperpipe to json
        :param pipe: Hyperpipe to transform
        :return: dict representation of hyperpipe
        """
        self.json = {}
        self.json["name"] = pipe.name
        for key in ["verbosity", "permutation_id", "cache_folder", "nr_of_processes"]:
            self.json[key] = getattr(pipe, key)
        self.json["random_seed"] = pipe.random_state

        self.json["inner_cv"] = self.transform_elements_recursive(pipe.cross_validation.inner_cv)
        self.json["outer_cv"] = self.transform_elements_recursive(pipe.cross_validation.outer_cv)
        for c_key in ["calculate_metrics_across_folds", "eval_final_performance", "test_size",
                      "calculate_metrics_per_fold"]:
            self.json[c_key] = getattr(pipe.cross_validation, c_key)

        self.json["performance_constraints"] = pipe.optimization.performance_constraints
        self.json["optimizer"] = pipe.optimization.optimizer_input_str
        if pipe.optimization.optimizer_params:
            self.json["optimizer_params"] = self.transform_elements_recursive(pipe.optimization.optimizer_params)
        self.json["metrics"] = self.transform_elements_recursive(pipe.optimization.metrics)
        self.json["best_config_metric"] = pipe.optimization.best_config_metric

        if pipe.output_settings:
            self.json["output_settings"] = {"mongodb_connect_url" : pipe.output_settings.mongodb_connect_url,
                                            "save_output" : pipe.output_settings.save_output,
                                            "plots" : pipe.output_settings.plots,
                                            "overwrite_results" : pipe.output_settings.overwrite_results,
                                            "project_folder" : pipe.output_settings.project_folder,
                                            "user_id" : pipe.output_settings.user_id,
                                            "wizard_object_id" : pipe.output_settings.wizard_object_id,
                                            "wizard_project_name" : pipe.output_settings.wizard_project_name,
                                            "__photon_type" : "OutputSettings"}

        if pipe.preprocessing:
            self.json["preprocessing"] = {"elements" : self.transform_elements_recursive(pipe.preprocessing.elements),
                                         "__photon_type": "Preprocessing"
                                          }

        self.json["elements"] = self.transform_elements_recursive(pipe.elements)

        return self.json

    def transform_elements_recursive(self, element):
        """
        recursive functinon for Hyperpipe.elements and all class types.
        :param element: as type dict, list, tuple, str, bool, ...
        :return: json like version
        """
        d = {}
        if element is None:
            return d
        # main dtype
        if any(isinstance(element, t) for t in [int, bool, str, float, np.float]):
            return element
        # dtype == list or dtype == tuple
        if isinstance(element, list) or isinstance(element, tuple):
            tmp_list = []
            for e in element:
                tmp_list.append(self.transform_elements_recursive(e))
            if isinstance(element, list):
                return tmp_list
            else:
                return tuple(tmp_list)
        # dtype == dict
        elif isinstance(element, dict):
            for k,v in element.items():
                if not (k.startswith('_') or v is None):
                    if str(element.__class__.__name__) in self.attribute_allocator.keys():
                        if k in self.attribute_allocator[str(element.__class__.__name__)]:
                            d[k] = self.transform_elements_recursive(v)
                    elif k not in self.black_list:
                        d[k] = self.transform_elements_recursive(v)
        # dtype == object
        else:
            for k,v in (dict(inspect.getmembers(element))).items():
                if not (k.startswith('_') or v is None or inspect.ismethod(v)):
                    if str(element.__class__.__name__) in self.attribute_allocator.keys():
                        if k in self.attribute_allocator[str(element.__class__.__name__)]:
                                d[k] = self.transform_elements_recursive(v)
                    elif k not in self.black_list:
                        d[k] = self.transform_elements_recursive(v)
        if d and str(element.__class__.__name__) not in ["dict", "type", "property"]:
            d['__photon_type'] = str(element.__class__.__name__)
            if d['__photon_type'] in ["IntegerRange", "FloatRange", "NumberType", "BooleanSwitch"] \
                    and 'values' in d:
                del d['values']
        d = {key:val for key, val in d.items() if val is not None}
        if not d:
            return None
        return d

    def from_json_file(self, path: str):
        """
        main function to load from file.json
        :param path: storage path
        :return: json value
        """
        self.json = self.read_json_file(path)
        return self.from_json(self.json)

    def from_json(self, json):
        """
        manual to read hyperpipe json
        :param json: sotrage json
        :return: PHOTON.Hyperpipe
        """
        self.json = json
        for key in ['inner_cv', 'outer_cv', 'output_settings']:
            self.json[key] = self.load_elements_recursive(self.json[key])
        init = {key:value for key, value in self.json.items() if key not in ["elements", "preprocessing"]}
        pipe = self.str_to_class("Hyperpipe")(**init)
        elements = self.load_elements_recursive(self.json["elements"])
        if "preprocessing" in json:
            pre_elements = self.load_elements_recursive(self.json["preprocessing"]["elements"])
            pre = Preprocessing()
            for pre_element in pre_elements:
                pre += pre_element
            pipe += pre
        for element in elements:
            pipe += element
        return pipe

    def load_elements_recursive(self, json):
        """
        reverse of transform_elements_recursive
        :param json: dict, list, tuple, str, bool, ...
        :return: python class/type
        """
        a = {}
        b = []
        if json is None:
            return None
        # main dtypes
        if any(isinstance(json, t) for t in [int, bool, str, float]):
            return json
        # dtype == list
        if isinstance(json, list):
            for element in json:
                b.append(self.load_elements_recursive(element))
            return b
        # dtype == dict
        elif isinstance(json, dict):
            for element in json.keys():
                if element not in ["initial_name", "__photon_type", "kwargs", "initial_hyperparameters"]:
                    a.update({element: self.load_elements_recursive(json[element])})
                elif element == "kwargs":
                    for key in json["kwargs"].keys():
                        a.update({key: self.load_elements_recursive(json[element][key])})
                elif element == "initial_hyperparameters":
                    tmp = {}
                    for key in json[element].keys():
                        if key == "__photon_type":
                            continue
                        tmp.update({key: self.load_elements_recursive(json[element][key])})
                    a["hyperparameters"] = tmp
                elif element == "initial_name":
                    a["name"] = json['initial_name']
            if "__photon_type" in a:
                del a["__photon_type"]
            return self.str_to_class(json[element])(**a)
