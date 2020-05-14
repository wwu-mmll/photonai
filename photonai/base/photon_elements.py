import importlib
import importlib.util
import inspect
from copy import deepcopy
import os
from os import path
import json
import sys

from dataclasses import dataclass
from typing import List, ClassVar, Union, Dict, Any  # Hashable, Callable,
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import ParameterGrid
from sklearn.datasets import load_breast_cancer, load_boston

from photonai.base.photon_pipeline import PhotonPipeline
from photonai.helper.helper import PhotonDataHelper
from photonai.optimization.config_grid import (
    create_global_config_grid,
    create_global_config_dict,
)
from photonai.processing.metrics import Scorer
from photonai.photonlogger.logger import logger
from photonai.errors import raise_PhotonaiError

# class ElementDictionary:
#     """
#     Attributes
#     ----------
#     """
#    PHOTON_REGISTRIES = ["PhotonCore", "PhotonCluster", "PhotonNeuro"]


@dataclass
class PhotonRegistry:
    """
    Helper class to manage the PHOTON Element Register.

    Use it to add and remove items into the register.
    You can also retrieve information about items and its hyperparameters.

    Every item in the register is encoded by a string literal that points to a python class and its namespace.
    You can access the python class via the string literal.
    The class PhotonElement imports and instantiates the class for you.

    There is a distinct json file with the elements registered for each photon package (core, neuro, genetics, ..)
    There is also a json file for the user's custom elements.

    element data structure
    key   [  package, sklearn-pipeline-type]
    "PCA": [ "sklearn.decomposition.PCA", "Transformer"]

    Example
    -------
        from photonai.configuration.Register import PhotonRegister

        # get info about object, name, namespace and possible hyperparameters
        PhotonRegister.info("SVC")

        # show all items that are registered
        PhotonRegister.list()

        # register new object
        PhotonRegister.register("ABC1", "namespace.filename.ABC1", "Transformer")

        # delete it again.
        PhotonRegister.delete("ABC1")

    """

    custom_elements_folder: str = None
    custom_elements: str = None
    custom_elements_file: str = None
    base_PHOTON_REGISTRIES: ClassVar[List[str]] = [
        "PhotonCore",
        "PhotonCluster",
        "PhotonNeuro",
    ]
    PHOTON_REGISTRIES: ClassVar[List[str]] = [
        "PhotonCore",
        "PhotonCluster",
        "PhotonNeuro",
    ]

    EM_PKG_ID: ClassVar[int] = 0
    EM_SK_TYPE_ID: ClassVar[int] = 1
    EMD_OUT_OF_BOUNDS: ClassVar[int] = 2

    def __post_init__(self):
        if self.custom_elements_folder:
            self._load_custom_folder(self.custom_elements_folder)

    # # BHC 1.1
    def reset(self):
        #
        """
        start over to initial state,
        Set instance variable to class global variable
        """
        PhotonRegistry.PHOTON_REGISTRIES = PhotonRegistry.base_PHOTON_REGISTRIES

    @staticmethod
    def load_json(photon_package: str) -> Any:
        """
        load_json Loads JSON file.

        The class init PipelineElement('name',...)
        stores the element metadata in a json file.

        The JSON files are stored in the framework folder
        by the name convention 'photon_<package>.json'.
        (example:$HOME/PROJECTS/photon/photonai/base/registry/PhotonCore.json)

        The file is of format
        { name-1: ['import-pkg-class-path-1', class-path-1)],
          name-2: ['import-pkg-class-path-2', class-path-2)],
         ....}

        Parameters
        ----------
            photon_package:  The name of the photonai package of element metadata
        Returns
        -------
            [file_content, file_name]
        Notes
        -------
            if  JSON file does not exist, then create blank one.
        """

        folder = (
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            + "/registry"
        )

        file_name = os.path.join(folder, photon_package + ".json")
        if os.path.isfile(file_name):
            # Reading json
            with open(file_name, "r") as f:
                file_content = json.load(f)
        else:
            file_content = dict()
            with open(file_name, "w") as f:
                json.dump(file_content, f)

        #        return file_content, file_name
        return file_content

    @staticmethod
    def elements(element_metadata: Dict) -> List:
        return element_metadata.keys()

    @staticmethod
    def get_element_metadata(element_name: str, element_metadata: Dict) -> List:

        """
        Unstuture element metadata.
        Current form

        element-name: [class-path sklearn-api-type ('transformer' | 'estimator']

        Parameters
        ----------
        name
        element_metadata

        Returns
        -------

        """
        if element_name in element_metadata:
            element_pkg = element_metadata[element_name][PhotonRegistry.EM_PKG_ID]
            def strip_element_name(element_name: str, element_pkg: str) -> str:
                return element_pkg.replace(('.' + element_name), '')

            element_pkg = strip_element_name(element_name, element_pkg)

            element_imported_module = __import__(
                element_pkg, globals(), locals(), element_name, 0
            )
            return (
                element_name,
                element_pkg,
                element_imported_module,
                element_metadata[element_name][PhotonRegistry.EM_SK_TYPE_ID],
            )
        else:
            raise_PhotonaiError(
                "Element: {} of {} not found ".format(
                    name, list(element_metadata.keys())
                )
            )

    @staticmethod
    def get_package_info(photon_package: list = []) -> dict:
        """
        Collect all registered elements from JSON file

        Parameters:
        -----------
        * 'photon_package' [list]:
          The names of the PHOTON submodules for which the elements should be retrieved

        Returns
        -------
        Dict of registered elements
        """
        if photon_package == []:
            photon_package = PhotonRegistry.PHOTON_REGISTRIES
        element_metadata = dict()

        for package in photon_package:
            #            element_metadata, _ = ElementDictionary.load_json(package)
            element_metadata.update(PhotonRegistry.load_json(package))

        return element_metadata

        #     for key in element_metadata:
        #         class_path, class_name = os.path.splitext(element_metadata[key][0])
        #         class_info[key] = class_path, class_name[1:]
        # return class_info

    def _load_custom_folder(self, custom_elements_folder):
        self.custom_elements_folder, self.custom_elements_file = self._check_custom_folder(
            custom_elements_folder
        )

        # load custom elements from json
        self.custom_elements = self._load_json("CustomElements")
        PhotonRegistry.PHOTON_REGISTRIES.append("CustomElements")

    @staticmethod
    def _check_custom_folder(custom_folder):
        if not path.exists(custom_folder):
            logger.info("Creating folder {}".format(custom_folder))
            os.makedirs(custom_folder)

        custom_file = path.join(custom_folder, "CustomElements.json")
        if not path.isfile(custom_file):
            logger.info("Creating CustomElements.json")
            with open(custom_file, "w") as f:
                json.dump("", f)

        return custom_folder, custom_file

    def activate(self):
        if not self.custom_elements_folder:
            raise ValueError(
                "To activate a custom elements folder, specify a folder when instantiating the registry "
                "module. Example: registry = PhotonRegistry('/MY/CUSTOM/ELEMENTS/FOLDER) "
                "In case you don't have any custom models, there is no need to activate the registry."
            )
        if not path.exists(self.custom_elements_folder):
            raise FileNotFoundError(
                "Couldn't find custom elements folder: {}".format(
                    self.custom_elements_folder
                )
            )
        if not path.isfile(
            path.join(self.custom_elements_folder, "CustomElements.json")
        ):
            raise FileNotFoundError(
                "Couldn't find CustomElements.json. Did you register your element first?"
            )

        # add folder to python path
        logger.info("Adding custom elements folder to system path...")
        sys.path.append(self.custom_elements_folder)

        PipelineElement.ELEMENT_DICTIONARY.update(
            self._get_package_info(["CustomElements"])
        )
        logger.info("Successfully activated custom elements!")

    # 1.1 BHC register from call rather than file
    def register(self, photon_name: str, class_str: str, element_type: str):
        """
        Save element information to the JSON file

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal with which you want to access the class
        * 'class_str' [str]:
          The namespace of the class, like in the import statement
        * 'ml_type' [str]:
          Can be 'Estimator' or 'Transformer'
        * 'custom_folder' [str]:
          All registrations are saved to this folder
        """

        if not Scorer.is_element_type(element_type):
            raise ValueError(
                "Variable element_type must be {}, was: {}".format(
                    Scorer.ELEMENT_TYPES, element_type
                )
            )

        # check if folder exists
        if not self.custom_elements_folder:
            raise ValueError(
                "To register an element, specify a custom elements folder when instantiating the registry "
                "module. Example: registry = PhotonRegistry('/MY/CUSTOM/ELEMENTS/FOLDER)"
            )

        duplicate = self._check_duplicate(
            photon_name=photon_name, class_str=class_str, content=self.custom_elements
        )

        if not duplicate:
            python_file = path.join(
                self.custom_elements_folder, class_str.split(".")[0] + ".py"
            )
            if not path.isfile(python_file):
                raise FileNotFoundError(
                    "Couldn't find python file {} in your custom elements folder. "
                    "Please copy your file into this folder first!".format(python_file)
                )
            # add new element
            self.custom_elements[photon_name] = class_str, element_type

            # write back to file
            self._write2json(self.custom_elements)
            logger.info(
                "Adding PipelineElement "
                + class_str
                + ' to CustomElements.json as "'
                + photon_name
                + '".'
            )

            # activate custom elements
            self.activate()

            # check custom element
            logger.info("Running tests on custom element...")
            return self._run_tests(photon_name, element_type)
        else:
            logger.error("Could not register element!")

    def _run_tests(self, photon_name, element_type):
        # check import
        custom_element = PipelineElement(photon_name)

        # check if has fit, transform, predict
        if not hasattr(custom_element.base_element, "fit"):
            raise NotImplementedError("Custom element does not implement fit() method.")

        if element_type == "Transformer" and not hasattr(
            custom_element.base_element, "transform"
        ):
            raise NotImplementedError(
                "Custom element does not implement transform() method."
            )

        if element_type == "Estimator" and not hasattr(
            custom_element.base_element, "predict"
        ):
            raise NotImplementedError(
                "Custom element does not implement predict() method."
            )

        # check if estimator is regressor or classifier
        if element_type == "Estimator":
            if hasattr(custom_element.base_element, "_estimator_type"):
                est_type = getattr(custom_element.base_element, "_estimator_type")
                if est_type == "regressor":
                    X, y = load_boston(True)
                elif est_type == "classifier":
                    X, y = load_breast_cancer(True)
                else:
                    raise ValueError(
                        "Custom element does not specify whether it is a regressor or classifier. "
                        "Is {}".format(est_type)
                    )
            else:
                raise NotImplementedError(
                    "Custom element does not specify whether it is a regressor or classifier. "
                    "Consider inheritance from ClassifierMixin or RegressorMixin or set "
                    "_estimator_type manually."
                )
        else:
            X, y = load_boston(True)

        # try and test functionality
        kwargs = {"covariates": np.random.randn(len(y))}

        try:
            # test fit
            returned_element = custom_element.base_element.fit(X, y, **kwargs)
        except Exception as e:
            logger.info(
                "Not able to run tests on fit() method. Test data not compatible."
            )
            return e

        if not isinstance(returned_element, custom_element.base_element.__class__):
            raise NotImplementedError("fit() method does not return self.")

        try:
            # test transform or predict (if base element does not implement transform method, predict should be called
            # by PipelineElement -> so we only need to test transform()
            if custom_element.needs_y:
                if element_type == "Estimator":
                    raise NotImplementedError("Estimator should not need y.")
                Xt, yt, kwargst = custom_element.base_element.transform(X, y, **kwargs)
                if "covariates" not in kwargst.keys():
                    raise ValueError(
                        "Custom element does not correctly transform kwargs although needs_y is True. "
                        "If you change the number of samples in transform(), make sure to transform kwargs "
                        "respectively."
                    )
                if not len(kwargst["covariates"]) == len(X):
                    raise ValueError(
                        "Custom element is not returning the correct number of samples!"
                    )

            elif custom_element.needs_covariates:
                if element_type == "Estimator":
                    yt, kwargst = custom_element.base_element.predict(X, **kwargs)
                    if not len(yt) == len(y):
                        raise ValueError(
                            "Custom element is not returning the correct number of samples!"
                        )
                else:
                    Xt, kwargst = custom_element.base_element.transform(X, **kwargs)

                    if not len(Xt) == len(X) or not len(kwargst["covariates"]) == len(
                        X
                    ):
                        raise ValueError(
                            "Custom element is not returning the correct number of samples!"
                        )

            else:
                if element_type == "Estimator":
                    yt = custom_element.base_element.predict(X)
                    if not len(yt) == len(y):
                        raise ValueError(
                            "Custom element is not returning the correct number of samples!"
                        )
                else:
                    Xt = custom_element.base_element.transform(X)
                    if not len(Xt) == len(X):
                        raise ValueError(
                            "Custom element is not returning the correct number of samples!"
                        )

        except ValueError as ve:
            if "too many values to unpack" in ve.args[0]:
                raise ValueError(
                    "Custom element does not return X, y and kwargs the way it should "
                    "according to needs_y and needs_covariates."
                )
            else:
                logger.info(ve.args[0])
                return ve
        except Exception as e:
            logger.info(e.args[0])
            logger.info(
                "Not able to run tests on transform() or predict() method. Test data probably not compatible."
            )
            return e

        logger.info("All tests on custom element passed.")

    def info(self, photon_name: str, verbose: bool=1) -> Union[bool, List]:
        """
        Show information for object that is encoded by this name.

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal which accesses the class
        Returns
        -------
        (element_name, element_namespace, args_dict)
        """
        element_metadata = self._get_package_info()
        element_name, element_pkg, element_imported_module, _ =\
            PhotonRegistry.get_element_metadata(
            photon_name, element_metadata
        )

        if photon_name in element_metadata:
            element_namespace, element_name = element_metadata[photon_name]

            if verbose:
                print("----------------------------------")
                print("Name: " + element_name)
                print("Namespace: " + element_pkg)
                print("----------------------------------")

            try:

                desired_class = getattr(element_imported_module, element_name)
                base_element = desired_class()
                print("Possible Hyperparameters as derived from constructor:")
                class_args = inspect.signature(base_element.__init__)
                arg_d = {}
                for item, more_info in class_args.parameters.items():
                    arg_d[item] = item
                    if verbose: print("{:<35} {:<75}".format(item, str(more_info)))
                if verbose: print("----------------------------------")
                return element_name, element_namespace, arg_d
            except Exception as e:
                logger.error(e)
                logger.error(
                    "Could not instantiate class "
                    + element_namespace
                    + "."
                    + element_name
                )
        else:
            logger.error("Could not find element " + photon_name)
        return False

    def delete(self, photon_name):
        """
        Delete Element from JSON file

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal encoding the class
        """

        if photon_name in self.custom_elements:
            del self.custom_elements[photon_name]

            self._write2json(self.custom_elements)
            logger.info(
                'Removing the PipelineElement named "{0}" from CustomElements.json.'.format(
                    photon_name
                )
            )
        else:
            logger.info(
                'Cannot remove "{0}" from CustomElements.json. Element has not been registered before.'.format(
                    photon_name
                )
            )

    @staticmethod
    def _check_duplicate(photon_name, class_str, content):
        """
        Helper function to check if the entry is either registered by a different name or if the name is already given
        to another class

         Parameters:
        -----------
        * 'content':
          The content of the CustomElements.json
        * 'class_str' [str]:
          The namespace.Classname, where the class lives, from where it should be imported.
        * 'photon_name':
          The name of the element with which it is called within PHOTON
        Returns:
        --------
        Bool, False if there is no key with this name and the class is not already registered with another key
        """

        # check for duplicate name (dict key)
        if photon_name in content:
            logger.info(
                "A PipelineElement named "
                + photon_name
                + " has already been registered."
            )
            return True

        # check for duplicate class_str
        if any(class_str in ".".join([s[0], s[1]]) for s in content.values()):
            logger.info(
                "The Class named " + class_str + " has already been registered."
            )
            return True
        return False

    def _load_json(self, photon_package: str):
        """
        Load JSON file in which the elements for the PHOTON submodule are stored.

        The JSON files are stored in the framework folder by the name convention 'photon_package.json'

        Parameters:
        -----------
        * 'photon_package' [str]:
          The name of the photonai submodule

        Returns:
        --------
        JSON file as dict, file path as str
        """

        if photon_package == "CustomElements":
            folder = self.custom_elements_folder
            if not folder:
                return {}
        else:
            folder = (
                os.path.dirname(
                    os.path.abspath(inspect.getfile(inspect.currentframe()))
                )
                + "/registry/"
            )

        file_name = path.join(folder, photon_package + ".json")
        file_content = {}

        # Reading json
        with open(file_name, "r") as f:
            try:
                file_content = json.load(f)
            except json.JSONDecodeError as jde:
                # handle empty file
                if jde.msg == "Expecting value":
                    logger.error("Package File " + file_name + " was empty.")
                else:
                    raise jde
        if not file_content:
            file_content = dict()
        return file_content

    def _write2json(self, content2write: dict):
        """
        Write json content to file

        Parameters:
        -----------
        * 'content2write' [dict]:
          The new information to attach to the file
        * 'photon_package' [str]:
          The PHOTON submodule name to which the new class belongs, so it is written to the correct json file
        """
        # Writing JSON data
        with open(self.custom_elements_file, "w") as f:
            json.dump(content2write, f)

    def _get_package_info(self, photon_package: list = PHOTON_REGISTRIES) -> dict:
        """
        Collect all registered elements from JSON file

        Parameters:
        -----------
        * 'photon_package' [list]:
          The names of the PHOTON submodules for which the elements should be retrieved

        Returns
        -------
        Dict of registered elements
        """
        class_info = dict()
        for package in photon_package:

            content = self._load_json(package)

            for key in content:
                class_path, class_name = os.path.splitext(content[key][0])
                class_info[key] = class_path, class_name[1:]
        return class_info

    def list_available_elements(self, photon_package=PHOTON_REGISTRIES):
        """
        Print info about all items that are registered for the PHOTON submodule to the console.

        Parameters:
        -----------
        * 'photon_package' [list]:
          The names of the PHOTON submodules for which the elements should be retrieved
        """
        if isinstance(photon_package, str):
            photon_package = [photon_package]
        for package in photon_package:
            content = self._load_json(package)
            if len(content) > 0:
                print("\n" + package)
                for k, v in sorted(content.items()):
                    class_info, package_type = v
                    print("{:<35} {:<75} {:<5}".format(k, class_info, package_type))


class PhotonNative:
    """only for checking if code is meeting requirements"""

    pass


class PipelineElement(BaseEstimator):
    """
    Photon wrapper class for any transformer or predictor element in the pipeline.

    1. Saves the hyperparameters that are to be tested and creates a grid of all hyperparameter configurations
    2. Enables fast and rapid instantiation of pipeline elements per string identifier,
         e.g 'svc' creates an sklearn.svm.SVC object.
    3. Attaches a "disable" switch to every element in the pipeline in order to test a complete disable


    Parameters
    ----------
    * `name` [str]:
       A string literal encoding the class to be instantiated
    * `hyperparameters` [dict]:
       Which values/value range should be tested for the hyperparameter.
       In form of "Hyperparameter_name: [array of parameter values to be tested]"
    * `test_disabled` [bool]:
        If the hyperparameter search should evaluate a complete disabling of the element
    * `disabled` [bool]:
        If true, the element is currently disabled and does nothing except return the data it received
    * `kwargs` [dict]:
        Any parameters that should be passed to the object to be instantiated, default parameters

    """

    # Registering Pipeline Elements
    ELEMENT_DICTIONARY = PhotonRegistry.get_package_info()

    def __init__(
        self,
        name,
        hyperparameters: dict = None,
        test_disabled: bool = False,
        disabled: bool = False,
        base_element=None,
        batch_size=0,
        **kwargs
    ):
        """
        Takes a string literal and transforms it into an object
        of the associated class (see PhotonCore.JSON)

        Returns
        -------
        instantiated class object
        """
        if hyperparameters is None:
            hyperparameters = {}

        element_metadata =  PipelineElement.ELEMENT_DICTIONARY
        if base_element is None:
            if name in element_metadata:
                try:

                    #            desired_class_home. - = element_metadata(name)
                    #            desired_class_name = name
                    #            desired_class = getattr(imported_module, desired_class_name)
                    #            self.base_element = desired_class(**kwargs)
                    # desired_class_info = PipelineElement.ELEMENT_DICTIONARY[name]
                    # desired_class_home = desired_class_info[0]
                    # desired_class_name = name
                    #
                    # imported_module = __import__(
                    #     desired_class_home, globals(), locals(), desired_class_name, 0
                    # )
                    #                    imported_module = importlib.import_module(desired_class_home)
                    element_name, element_pkg, element_imported_module, _ = \
                        PhotonRegistry.get_element_metadata(
                            name, element_metadata
                        )
                    desired_class = getattr(element_imported_module, name)
                    self.base_element = desired_class(**kwargs)
                except AttributeError as ae:
                    logger.error(
                        "ValueError: Could not find according class:"
                        + str(PipelineElement.ELEMENT_DICTIONARY[name])
                    )
                    raise ValueError(
                        "Could not find according class:",
                        PipelineElement.ELEMENT_DICTIONARY[name],
                    )
            else:
                logger.error("Element not supported right now:" + name)
                raise NameError("Element not supported right now:", name)
        else:
            self.base_element = base_element

        self.is_transformer = hasattr(self.base_element, "transform")
        self.is_estimator = hasattr(self.base_element, "predict")
        self._name = name
        self.initial_name = str(name)
        self.kwargs = kwargs
        self.current_config = None
        self.batch_size = batch_size
        self.test_disabled = test_disabled
        self.initial_hyperparameters = dict(hyperparameters)

        self._sklearn_disabled = self.name + "__disabled"
        self._hyperparameters = hyperparameters
        if len(hyperparameters) > 0:
            key_0 = next(iter(hyperparameters))
            if self.name not in key_0:
                self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = hyperparameters
        # self.initalize_hyperparameters = hyperparameters
        # check if hyperparameters are already in sklearn style

        # check if hyperparameters are members of the class
        if self.is_transformer or self.is_estimator:
            self._check_hyperparameters(BaseEstimator)

        self.disabled = disabled

        # check if self.base element needs y for fitting and transforming
        if hasattr(self.base_element, "needs_y"):
            self.needs_y = self.base_element.needs_y
        else:
            self.needs_y = False
        # or if it maybe needs covariates for fitting and transforming
        if hasattr(self.base_element, "needs_covariates"):
            self.needs_covariates = self.base_element.needs_covariates
        else:
            self.needs_covariates = False

        self._random_state = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.generate_sklearn_hyperparameters(self.initial_hyperparameters)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: dict):
        self.generate_sklearn_hyperparameters(value)

    def _check_hyperparameters(self, BaseEstimator):
        # check if hyperparameters are members of the class
        not_supported_hyperparameters = list(
            set(
                [
                    key.split("__")[-1]
                    for key in self._hyperparameters.keys()
                    if key.split("__")[-1] != "disabled"
                ]
            )
            - set(BaseEstimator.get_params(self.base_element).keys())
        )
        if not_supported_hyperparameters:
            error_message = (
                "ValueError: Set of hyperparameters are not valid, check hyperparameters:"
                + str(not_supported_hyperparameters)
            )
            logger.error(error_message)
            raise ValueError(error_message)

    def generate_sklearn_hyperparameters(self, value: dict):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        """
        self._hyperparameters = {}
        for attribute, value_list in value.items():
            self._hyperparameters[self.name + "__" + attribute] = value_list
        if self.test_disabled:
            self._hyperparameters[self._sklearn_disabled] = [False, True]

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
        if hasattr(self, "elements"):
            for el in self.elements:
                if hasattr(el, "random_state"):
                    el.random_state = self._random_state
        if hasattr(self, "base_element") and hasattr(self.base_element, "random_state"):
            self.base_element.random_state = random_state

    @property
    def _estimator_type(self):
        if hasattr(self.base_element, "_estimator_type"):
            est_type = getattr(self.base_element, "_estimator_type")
            if est_type not in Scorer.ML_TYPES:
                raise NotImplementedError(
                    "Currently, we only support {}. Is {}.".format(
                        Scorer.ML_TYPES, est_type
                    )
                )
            if not hasattr(self.base_element, "predict"):
                raise NotImplementedError(
                    "Estimator does not implement predict() method."
                )
            return est_type
        else:
            if hasattr(self.base_element, "predict"):
                raise NotImplementedError(
                    "Element has predict() method but does not specify whether it is a regressor "
                    "or classifier. Remember to inherit from ClassifierMixin or RegressorMixin."
                )
            else:
                return None

    # this is only here because everything inherits from PipelineElement.
    def __iadd__(self, pipe_element):
        """
        Add an element to the element list
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement or Hyperpipe]:
            The object to add, being either a transformer or an estimator.

        """
        PipelineElement.sanity_check_element_type_for_building_photon_pipes(
            pipe_element, type(self)
        )

        # check if that exact instance has been added before
        already_added_objects = len([i for i in self.elements if i is pipe_element])
        if already_added_objects > 0:
            error_msg = (
                "Cannot add the same instance twice to "
                + self.name
                + " - "
                + str(type(self))
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # check for doubled names:
        already_existing_element_with_that_name = len(
            [i for i in self.elements if i.name == pipe_element.name]
        )

        if already_existing_element_with_that_name > 0:
            error_msg = (
                "Already added a pipeline element with the name "
                + pipe_element.name
                + " to "
                + self.name
            )
            logger.warn(error_msg)

            # check for other items that have been renamed
            nr_of_existing_elements_with_that_name = len(
                [i for i in self.elements if i.name.startswith(pipe_element.name)]
            )
            new_name = pipe_element.name + str(
                nr_of_existing_elements_with_that_name + 1
            )
            while len([i for i in self.elements if i.name == new_name]) > 0:
                nr_of_existing_elements_with_that_name += 1
                new_name = pipe_element.name + str(
                    nr_of_existing_elements_with_that_name + 1
                )

            logger.warn(
                "Renaming "
                + pipe_element.name
                + " in "
                + self.name
                + " to "
                + new_name
                + " in "
                + self.name
            )
            pipe_element.name = new_name

        self.elements.append(pipe_element)
        return self

    def copy_me(self):
        if self.name in self.ELEMENT_DICTIONARY:
            # we need initial name to refer to the class to be instantiated  (SVC) even though the name might be SVC2
            copy = PipelineElement(
                self.initial_name,
                {},
                test_disabled=self.test_disabled,
                disabled=self.disabled,
                batch_size=self.batch_size,
                **self.kwargs
            )
            copy.initial_hyperparameters = self.initial_hyperparameters
            # in the setter of the name, we use initial hyperparameters to adjust the hyperparameters to the name
            copy.name = self.name
        else:
            if hasattr(self.base_element, "copy_me"):
                new_base_element = self.base_element.copy_me()
            else:
                try:
                    new_base_element = deepcopy(self.base_element)
                except Exception as e:
                    error_msg = (
                        "Cannot copy custom element "
                        + self.name
                        + ". Please specify a copy_me() method "
                        "returning a copy of the object"
                    )
                    logger.error(error_msg)
                    raise e

            # handle custom elements
            copy = PipelineElement.create(
                self.name,
                new_base_element,
                hyperparameters=self.hyperparameters,
                test_disabled=self.test_disabled,
                disabled=self.disabled,
                batch_size=self.batch_size,
                **self.kwargs
            )
        if self.current_config is not None:
            copy.set_params(**self.current_config)
        copy._random_state = self._random_state
        return copy

    @classmethod
    def create(
        cls,
        name,
        base_element,
        hyperparameters: dict,
        test_disabled=False,
        disabled=False,
        **kwargs
    ):
        """
        Takes an instantiated object and encapsulates it into the PHOTON structure,
        add the disabled function and attaches information about the hyperparameters that should be tested
        """
        if isinstance(base_element, type):
            raise ValueError("Base element should be an instance but is a class.")
        return PipelineElement(
            name,
            hyperparameters,
            test_disabled,
            disabled,
            base_element=base_element,
            **kwargs
        )

    @property
    def feature_importances_(self):
        if hasattr(self.base_element, "feature_importances_"):
            return self.base_element.feature_importances_.tolist()
        elif hasattr(self.base_element, "coef_"):
            return self.base_element.coef_.tolist()

    def generate_config_grid(self):
        config_dict = create_global_config_dict([self])
        if len(config_dict) > 0:
            if self.test_disabled:
                config_dict.pop(self._sklearn_disabled)
            config_list = list(ParameterGrid(config_dict))
            if self.test_disabled:
                for item in config_list:
                    item[self._sklearn_disabled] = False
                config_list.append({self._sklearn_disabled: True})
                if len(config_list) < 2:
                    config_list.append({self._sklearn_disabled: False})

            return config_list
        else:
            return []

    def get_params(self, deep: bool = True):
        """
        Forwards the get_params request to the wrapped base element
        """
        if hasattr(self.base_element, "get_params"):
            params = self.base_element.get_params(deep)
            params["name"] = self.name
            return params
        else:
            return None

    def set_params(self, **kwargs):
        """
        Forwards the set_params request to the wrapped base element
        Takes care of the disabled parameter which is additionally attached by the PHOTON wrapper
        """
        # this is an ugly hack to approximate the right settings when copying the element
        self.current_config = kwargs
        # element disable is a construct used for this container only
        if self._sklearn_disabled in kwargs:
            self.disabled = kwargs[self._sklearn_disabled]
            del kwargs[self._sklearn_disabled]
        elif "disabled" in kwargs:
            self.disabled = kwargs["disabled"]
            del kwargs["disabled"]
        self.base_element.set_params(**kwargs)
        return self

    def fit(self, X, y=None, **kwargs):
        """
        Calls the fit function of the base element

        Returns
        ------
        self
        """
        if not self.disabled:
            obj = self.base_element
            arg_list = inspect.signature(obj.fit)
            if len(arg_list.parameters) > 2:
                vals = arg_list.parameters.values()
                kwargs_param = list(vals)[-1]
                if kwargs_param.kind == kwargs_param.VAR_KEYWORD:
                    obj.fit(X, y, **kwargs)
                    return self
            obj.fit(X, y)
        return self

    def __batch_predict(self, delegate, X, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            logger.warn("Cannot do batching on a single entity.")
            return delegate(X, **kwargs)

            # initialize return values
        processed_y = None
        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1
            logger.debug(self.name + " is predicting batch " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(
                X, None, kwargs, start, stop
            )

            # predict
            y_pred = delegate(X_batched, **kwargs_dict_batched)
            processed_y = PhotonDataHelper.stack_data_vertically(processed_y, y_pred)

        return processed_y

    def __predict(self, X, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, "predict"):
                # Todo: check if element has kwargs, and give it to them
                # todo: so this todo above was old, here are my changes:
                # return self.base_element.predict(X)
                return self.adjusted_predict_call(
                    self.base_element.predict, X, **kwargs
                )
            else:
                logger.error(
                    "BaseException. base Element should have function " + "predict."
                )
                raise BaseException("base Element should have function predict.")
        else:
            return X

    def predict(self, X, **kwargs):
        """
        Calls predict function on the base element.
        """
        if self.batch_size == 0:
            return self.__predict(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict, X, **kwargs)

    def predict_proba(self, X, **kwargs):
        if self.batch_size == 0:
            return self.__predict_proba(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict_proba, X, **kwargs)

    def __predict_proba(self, X, **kwargs):
        """
        Predict probabilities
        base element needs predict_proba() function, otherwise throw
        base exception.
        """
        if not self.disabled:
            if hasattr(self.base_element, "predict_proba"):
                # todo: here, I used delegate call (same as below in predict within the transform call)
                # return self.base_element.predict_proba(X)
                return self.adjusted_predict_call(
                    self.base_element.predict_proba, X, **kwargs
                )
            else:

                # todo: in case _final_estimator is a Branch, we do not know beforehand it the base elements will
                #  have a predict_proba -> if not, just return None (@Ramona, does this make sense?)
                # logger.error('BaseException. base Element should have "predict_proba" function.')
                # raise BaseException('base Element should have predict_proba function.')
                return None
        return X

    def __transform(self, X, y=None, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, "transform"):
                return self.adjusted_delegate_call(
                    self.base_element.transform, X, y, **kwargs
                )
            elif hasattr(self.base_element, "predict"):
                return self.predict(X, **kwargs), y, kwargs
            else:
                logger.error("BaseException: transform-predict-mess")
                raise BaseException("transform-predict-mess")
        else:
            return X, y, kwargs

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on the base element.

        IN CASE THERE IS NO TRANSFORM METHOD, CALLS PREDICT.
        This is used if we are using an estimator as a preprocessing step.
        """
        if self.batch_size == 0:
            return self.__transform(X, y, **kwargs)
        else:
            return self.__batch_transform(X, y, **kwargs)

    def inverse_transform(self, X, y=None, **kwargs):
        if hasattr(self.base_element, "inverse_transform"):
            # todo: check this
            X, y, kwargs = self.adjusted_delegate_call(
                self.base_element.inverse_transform, X, y, **kwargs
            )
        return X, y, kwargs

    def __batch_transform(self, X, y=None, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            logger.warn("Cannot do batching on a single entity.")
            return self.__transform(X, y, **kwargs)

            # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(
                X, y, kwargs, start, stop
            )

            actual_batch_size = PhotonDataHelper.find_n(X_batched)
            logger.debug(
                self.name
                + " is transforming batch "
                + str(batch_idx)
                + " with "
                + str(actual_batch_size)
                + " items."
            )

            # call transform
            X_new, y_new, kwargs_new = self.adjusted_delegate_call(
                self.base_element.transform, X_batched, y_batched, **kwargs_dict_batched
            )

            # stack results
            processed_X, processed_y, processed_kwargs = PhotonDataHelper.join_data(
                processed_X, X_new, processed_y, y_new, processed_kwargs, kwargs_new
            )

        return processed_X, processed_y, processed_kwargs

    def adjusted_delegate_call(self, delegate, X, y, **kwargs):
        # Case| transforms X | needs_y | needs_covariates
        # -------------------------------------------------------
        #   1         yes        no           no     = transform(X) -> returns Xt

        # todo: case does not exist any longer

        #   2         yes        yes          no     = transform(X, y) -> returns Xt, yt

        #   3         yes        yes          yes    = transform(X, y, kwargs) -> returns Xt, yt, kwargst
        #   4         yes        no           yes    = transform(X, kwargs) -> returns Xt, kwargst
        #   5         no      yes or no      yes or no      = NOT ALLOWED

        # todo: we don't need to check for Switch, Stack or Branch since those classes define
        # needs_y and needs_covariates in their __init__()
        if self.needs_y:
            # if we dont have any target vector, we are in "predict"-mode although we are currently transforming
            # in this case, we want to skip the transformation and pass X, None and kwargs onwards
            # so basically, we skip all training_only elements
            # todo: I think, there's no way around this if we want to pass y and kwargs down to children of Switch and Branch
            if isinstance(self, (Switch, Branch, Preprocessing)):
                X, y, kwargs = delegate(X, y, **kwargs)
            else:
                if y is not None:
                    # todo: in case a method needs y, we should also always pass kwargs
                    #  i.e. if we change the number of samples, we also need to apply that change to all kwargs
                    # todo: talk to Ramona! Maybe we actually DO need this case
                    if self.needs_covariates:
                        X, y, kwargs = delegate(X, y, **kwargs)
                    else:
                        X, y = delegate(X, y)
        elif self.needs_covariates:
            X, kwargs = delegate(X, **kwargs)

        else:
            X = delegate(X)

        return X, y, kwargs

    def adjusted_predict_call(self, delegate, X, **kwargs):
        if self.needs_covariates:
            return delegate(X, **kwargs)
        else:
            return delegate(X)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calls the score function on the base element:
        Returns a goodness of fit measure or a likelihood of unseen data:

        Parameters
        ----------
        X_test
        y_test

        Returns
        -------

        """
        return self.base_element.score(X_test, y_test)

    def prettify_config_output(
        self, config_name: str, config_value, return_dict: bool = False
    ):
        """Make hyperparameter combinations human readable """
        if config_name == "disabled" and config_value is False:
            if return_dict:
                return {"disabled": False}
            else:
                return "disabled = False"
        else:
            if return_dict:
                return {config_name: config_value}
            else:
                return config_name + "=" + str(config_value)

    @staticmethod
    def sanity_check_element_type_for_building_photon_pipes(pipe_element, type_of_self):
        if (
            not isinstance(pipe_element, PipelineElement)
            and not isinstance(pipe_element, PhotonNative)
        ) or isinstance(pipe_element, Preprocessing):
            raise TypeError(
                str(type_of_self)
                + " only accepts PHOTON elements. Cannot add element of type "
                + str(type(pipe_element))
            )


class Branch(PipelineElement):
    """
     A substream of pipeline elements that is encapsulated e.g. for parallelization

     Parameters
     ----------
        * `name` [str]:
            Name of the encapsulated item and/or summary of the encapsulated element`s functions

        """

    def __init__(self, name, elements=None):

        super().__init__(
            name, {}, test_disabled=False, disabled=False, base_element=True
        )

        # in case any of the children needs y or covariates we need to request them
        self.needs_y = True
        self.needs_covariates = True
        self.elements = []
        self.has_hyperparameters = True
        self.skip_caching = True

        # needed for caching on individual level
        self.fix_fold_id = False
        self.do_not_delete_cache_folder = False

        # add elements
        if elements:
            for element in elements:
                self.add(element)

    def fit(self, X, y=None, **kwargs):
        self.base_element = Branch.sanity_check_pipeline(self.base_element)
        return super().fit(X, y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        if self._estimator_type == "classifier" or self._estimator_type == "regressor":
            return super().predict(X), y, kwargs
        return super().transform(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def __iadd__(self, pipe_element):
        """
        Add an element to the sub pipeline
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement or Hyperpipe]:
            The object to add, being either a transformer or an estimator.

        """
        super(Branch, self).__iadd__(pipe_element)
        self._prepare_pipeline()
        return self

    def add(self, pipe_element):
        """
           Add an element to the sub pipeline
           Returns self

           Parameters
           ----------
           * `pipe_element` [PipelineElement or Hyperpipe]:
               The object to add, being either a transformer or an estimator.

           """
        self.__iadd__(pipe_element)

    @staticmethod
    def prepare_photon_pipe(elements):
        pipeline_steps = list()
        for item in elements:
            pipeline_steps.append((item.name, item))
        return PhotonPipeline(pipeline_steps)

    @staticmethod
    def sanity_check_pipeline(pipe):
        if isinstance(pipe.elements[-1][1], CallbackElement):
            raise Warning(
                "Last element of pipeline cannot be callback element, would be mistaken for estimator. Removing it."
            )
            Logger().warn(
                "Last element of pipeline cannot be callback element, would be mistaken for estimator. Removing it."
            )
            del pipeline_steps[-1]
        return pipe

    def _prepare_pipeline(self):
        """ Generates sklearn pipeline with all underlying elements """

        self._hyperparameters = {
            item.name: item.hyperparameters
            for item in self.elements
            if hasattr(item, "hyperparameters")
        }

        if self.has_hyperparameters:
            self.generate_sklearn_hyperparameters()
        new_pipe = Branch.prepare_photon_pipe(self.elements)
        new_pipe._fix_fold_id = self.fix_fold_id
        new_pipe._do_not_delete_cache_folder = self.do_not_delete_cache_folder
        self.base_element = new_pipe

    def copy_me(self):
        new_copy_of_me = self.__class__(self.name)
        for item in self.elements:
            if hasattr(item, "copy_me"):
                copy_item = item.copy_me()
            else:
                copy_item = deepcopy(item)
            new_copy_of_me += copy_item
        if self.current_config is not None:
            new_copy_of_me.set_params(**self.current_config)
        new_copy_of_me._random_state = self._random_state
        return new_copy_of_me

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        return

    @property
    def _estimator_type(self):
        return getattr(self.elements[-1], "_estimator_type")

    def generate_config_grid(self):
        if self.has_hyperparameters:
            tmp_grid = create_global_config_grid(self.elements, self.name)
            return tmp_grid
        else:
            return []

    def generate_sklearn_hyperparameters(self):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        """
        self._hyperparameters = {}
        for element in self.elements:
            for attribute, value_list in element.hyperparameters.items():
                self._hyperparameters[self.name + "__" + attribute] = value_list

    def _check_hyper(self, BaseEstimator):
        pass

    @property
    def feature_importances_(self):
        if hasattr(self.elements[-1], "feature_importances_"):
            return getattr(self.elements[-1], "feature_importances_")


class Preprocessing(Branch):
    """
        If a preprocessing pipe is added to a PHOTON Hyperpipe, all transformers are applied to the data ONCE
        BEFORE cross validation starts in order to prepare the data.
        Every added element should be a transformer PipelineElement.
    """

    def __init__(self):
        super().__init__("Preprocessing")
        self.has_hyperparameters = False
        self.needs_y = True
        self.needs_covariates = True
        self._name = "Preprocessing"
        self.is_transformer = True
        self.is_estimator = False

    def __iadd__(self, pipe_element):
        """
        Add an element to the sub pipeline
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement]:
            The transformer object to add.

        """
        if hasattr(pipe_element, "transform"):
            super(Preprocessing, self).__iadd__(pipe_element)
            if len(pipe_element.hyperparameters) > 0:
                raise ValueError(
                    "A preprocessing transformer must not have any hyperparameter "
                    "because it is not part of the optimization and cross validation procedure"
                )

        else:
            raise ValueError("Pipeline Element must have transform function")
        return self

    def predict(self, data, **kwargs):
        raise Warning(
            "There is no predict function of the preprocessing pipe, it is a transformer only."
        )
        pass

    @property
    def _estimator_type(self):
        return


class Stack(PipelineElement):
    """
    Creates a vertical stacking/parallelization of pipeline items.

    The object acts as single pipeline element and encapsulates several vertically stacked other pipeline elements, each
    child receiving the same input data. The data is iteratively distributed to all children, the results are collected
    and horizontally concatenated.

    """

    def __init__(self, name: str, elements=None, use_probabilities: bool = False):
        """
        Creates a new Stack element.
        Collects all possible hyperparameter combinations of the children

        Parameters
        ----------
        * `name` [str]:
            Give the pipeline element a name
        * `elements` [list, optional]:
            List of pipeline elements that should run in parallel
        * `voting` [bool]:
            If true, the predictions of the encapsulated pipeline elements are joined to a single prediction
        """
        super(Stack, self).__init__(
            name,
            hyperparameters={},
            test_disabled=False,
            disabled=False,
            base_element=True,
        )

        self._hyperparameters = {}
        self.elements = list()
        if elements is not None:
            for item_to_stack in elements:
                self.__iadd__(item_to_stack)

        # todo: Stack should not be allowed to change y, only covariates
        self.needs_y = False
        self.needs_covariates = True
        self.use_probabilities = use_probabilities

    def __iadd__(self, item):
        """
        Adds a new element to the stack.
        Generates sklearn hyperparameter names in order to set the item's hyperparameters in the optimization process.

        * `item` [PipelineElement or Branch or Hyperpipe]:
            The Element that should be stacked and will run in a vertical parallelization in the original pipe.
        """
        self.check_if_needs_y(item)
        super(Stack, self).__iadd__(item)

        # for each configuration
        tmp_dict = dict(item.hyperparameters)
        for key, element in tmp_dict.items():
            self._hyperparameters[self.name + "__" + key] = tmp_dict[key]

        return self

    def check_if_needs_y(self, item):
        if isinstance(item, (Branch, Stack, Switch)):
            for child_item in item.elements:
                self.check_if_needs_y(child_item)
        elif isinstance(item, PipelineElement):
            if item.needs_y:
                raise NotImplementedError(
                    "Elements in Stack must not transform y because the number of samples in every "
                    "element of the stack might differ. Then, it will not be possible to concatenate those "
                    "data and target matrices. Please use the transformer that is using y before or after "
                    "the stack."
                )

    def add(self, item):
        self.__iadd__(item)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        pass

    def generate_config_grid(self):
        tmp_grid = create_global_config_grid(self.elements, self.name)
        return tmp_grid

    def get_params(self, deep=True):
        all_params = {}
        for element in self.elements:
            all_params[element.name] = element.get_params(deep)
        return all_params

    def set_params(self, **kwargs):
        """
        Find the particular child and distribute the params to it
        """
        spread_params_dict = {}
        for k, val in kwargs.items():
            splitted_k = k.split("__")
            item_name = splitted_k[0]
            if item_name not in spread_params_dict:
                spread_params_dict[item_name] = {}
            dict_entry = {"__".join(splitted_k[1::]): val}
            spread_params_dict[item_name].update(dict_entry)

        for name, params in spread_params_dict.items():
            missing_element = (name, params)
            for element in self.elements:
                if element.name == name:
                    element.set_params(**params)
                    missing_element = None
            if missing_element:
                raise ValueError(
                    "Couldn't set hyperparameter for element {} -> {}".format(
                        missing_element[0], missing_element[1]
                    )
                )
        return self

    def fit(self, X, y=None, **kwargs):
        """
        Calls fit iteratively on every child
        """
        for element in self.elements:
            # Todo: parallellize fitting
            element.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        if not self.use_probabilities:
            return self._predict(X, **kwargs)
        else:
            return self.predict_proba(X, **kwargs)

    def _predict(self, X, **kwargs):
        """
        Iteratively calls predict on every child.
        """
        # Todo: strategy for concatenating data from different pipes
        # todo: parallelize prediction
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict(X, **kwargs)
            predicted_data = PhotonDataHelper.stack_data_horizontally(
                predicted_data, element_transform
            )
        return predicted_data

    def predict_proba(self, X, y=None, **kwargs):
        """
        Predict probabilities for every pipe element and stack them together.
        """
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict_proba(X)
            if element_transform is None:
                element_transform = element.predict(X)
            predicted_data = PhotonDataHelper.stack_data_horizontally(
                predicted_data, element_transform
            )
        return predicted_data

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on every child.

        If the encapsulated child is a hyperpipe, also calls predict on the last element in the pipeline.
        """
        transformed_data = np.array([])
        for element in self.elements:
            # if it is a hyperpipe with a final estimator, we want to use predict:
            element_transform, _, _ = element.transform(X, y, **kwargs)
            transformed_data = PhotonDataHelper.stack_data_horizontally(
                transformed_data, element_transform
            )

        return transformed_data, y, kwargs

    def copy_me(self):
        ps = Stack(self.name)
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps.base_element = self.base_element
        ps._random_state = self._random_state
        return ps

    def inverse_transform(self, X, y=None, **kwargs):
        raise NotImplementedError(
            "Inverse Transform is not yet implemented for a Stacking Element in PHOTON"
        )

    @property
    def _estimator_type(self):
        return None

    def _check_hyper(self, BaseEstimator):
        pass

    @property
    def feature_importances_(self):
        return


class Switch(PipelineElement):
    """
    This class encapsulates several pipeline elements that belong at the same step of the pipeline,
    competing for being the best choice.

    If for example you want to find out if preprocessing A or preprocessing B is better at this position in the pipe.
    Or you want to test if a tree outperforms the good old SVM.

    ATTENTION: This class is a construct that may be convenient but is not suitable for any complex optimizations.
    Currently it only works for grid_search and the derived optimization strategies.
    USE THIS ONLY FOR RAPID PROTOTYPING AND PRELIMINARY RESULTS

    The class acts as if it is a single entity. Tt joins the hyperparamater combinations of each encapsulated element to
    a single, big combination grid. Each hyperparameter combination from that grid gets a number. Then the Switch
    object publishes the numbers to be chosen as the object's hyperparameter. When a new number is chosen from the
    optimizer, it internally activates the belonging element and sets the element's parameter to the hyperparameter
    combination. In that way, each of the elements is tested in all its configurations at the same position in the
    pipeline. From the outside, the process and the optimizer only sees one parameter of the Switch, that is
    the an integer indicating which item of the hyperparameter combination grid is currently active.

    """

    def __init__(self, name: str, elements: list = None):
        """
        Creates a new Switch object and generated the hyperparameter combination grid

        Parameters
        ----------
        * `name` [str]:
            How the element is called in the pipeline
        * `elements` [list, optional]:
            The competing pipeline elements
        * `_estimator_type:
            Used for validation purposes, either classifier or regressor

        """
        self._name = name
        self.sklearn_name = self.name + "__current_element"
        self._hyperparameters = {}
        self._current_element = (1, 1)
        self.pipeline_element_configurations = []
        self.base_element = None
        self.disabled = False
        self.test_disabled = False
        self.batch_size = 0

        self.needs_y = True
        self.needs_covariates = True
        # we assume we test models against each other, but only guessing
        self.is_estimator = True
        self.is_transformer = True
        self._random_state = False

        self.elements_dict = {}

        if elements:
            self.elements = elements
            self.generate_private_config_grid()
            for p_element in elements:
                self.elements_dict[p_element.name] = p_element
        else:
            self.elements = []

    def __iadd__(self, pipeline_element):
        """
        Add a new estimator or transformer object to the switch container. All items change positions during testing.

        Parameters
        ----------
        * `pipeline_element` [PipelineElement]:
            Item that should be tested against other competing elements at that position in the pipeline.
        """
        super(Switch, self).__iadd__(pipeline_element)
        self.elements_dict[pipeline_element.name] = pipeline_element
        self.generate_private_config_grid()
        return self

    def add(self, pipeline_element):
        """
        Add a new estimator or transformer object to the switch container. All items change positions during testing.

        Parameters
        ----------
        * `pipeline_element` [PipelineElement]:
            Item that should be tested against other competing elements at that position in the pipeline.
        """
        self.__iadd__(pipeline_element)

    @property
    def hyperparameters(self):
        # Todo: return actual hyperparameters of all pipeline elements??
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_private_config_grid(self):
        # reset
        self.pipeline_element_configurations = []

        # calculate anew
        hyperparameters = []
        # generate possible combinations for each item respectively - do not mix hyperparameters across items
        for i, pipe_element in enumerate(self.elements):
            # distinct_values_config = create_global_config([pipe_element])
            # add pipeline switch name in the config so that the hyperparameters can be set from other classes
            # pipeline switch will give the hyperparameters to the respective child
            # distinct_values_config_copy = {}
            # for config_key, config_value in distinct_values_config.items():
            #     distinct_values_config_copy[self.name + "__" + config_key] = config_value

            if hasattr(pipe_element, "generate_config_grid"):
                element_configurations = pipe_element.generate_config_grid()
                final_configuration_list = []
                if len(element_configurations) == 0:
                    final_configuration_list.append({})
                # else:
                for dict_item in element_configurations:
                    # copy_of_dict_item = {}
                    # for key, value in dict_item.items():
                    #     copy_of_dict_item[self.name + '__' + key] = value
                    final_configuration_list.append(dict(dict_item))

            self.pipeline_element_configurations.append(final_configuration_list)
            hyperparameters += [(i, nr) for nr in range(len(final_configuration_list))]

        self._hyperparameters = {self.sklearn_name: hyperparameters}

    @property
    def current_element(self):
        return self._current_element

    @current_element.setter
    def current_element(self, value):
        self._current_element = value
        self.base_element = self.elements[self.current_element[0]]

    def get_params(self, deep: bool = True):
        if self.base_element:
            return self.base_element.get_params(deep)
        else:
            return {}

    def set_params(self, **kwargs):

        """
        The optimization process sees the amount of possible combinations and chooses one of them.
        Then this class activates the belonging element and prepared the element with the particular chosen configuration.

        """

        config_nr = None
        config = None

        # in case we are operating with grid search
        if self.sklearn_name in kwargs:
            config_nr = kwargs[self.sklearn_name]
        elif "current_element" in kwargs:
            config_nr = kwargs["current_element"]

        # in case we are operating with another optimizer
        if config_nr is None:

            # we need to identify the element to activate by checking for which element the optimizer gave params
            if kwargs is not None:
                config = kwargs
                # ugly hack because subscription is somehow not possible, we use the for loop but break
                for kwargs_key, kwargs_value in kwargs.items():
                    first_element_name = kwargs_key.split("__")[0]
                    self.base_element = self.elements_dict[first_element_name]
                    break
        else:
            if not isinstance(config_nr, (tuple, list)):
                logger.error("ValueError: current_element must be of type Tuple")
                raise ValueError("current_element must be of type Tuple")

            # grid search hack
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]

        if config:
            # remove name
            unnamed_config = {}
            for config_key, config_value in config.items():
                key_split = config_key.split("__")
                unnamed_config["__".join(key_split[1::])] = config_value
            self.base_element.set_params(**unnamed_config)
        return self

    def copy_me(self):

        ps = Switch(self.name)
        ps._random_state = self._random_state
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps._current_element = self._current_element
        return ps

    def prettify_config_output(self, config_name, config_value, return_dict=False):

        """
        Makes the sklearn configuration dictionary human readable

        Returns
        -------
        * `prettified_configuration_string` [str]:
            configuration as prettified string or configuration as dict with prettified keys
        """

        if isinstance(config_value, tuple):
            output = self.pipeline_element_configurations[config_value[0]][
                config_value[1]
            ]
            if not output:
                if return_dict:
                    return {self.elements[config_value[0]].name: None}
                else:
                    return self.elements[config_value[0]].name
            else:
                if return_dict:
                    return output
                return str(output)
        else:
            return super(Switch, self).prettify_config_output(config_name, config_value)

    def predict_proba(self, X, **kwargs):
        """
        Predict probabilities
        base element needs predict_proba() function, otherwise throw
        base exception.
        """
        if not self.disabled:
            if hasattr(self.base_element.base_element, "predict_proba"):
                return self.base_element.predict_proba(X)
            else:
                return None
        return X

    def _check_hyper(self, BaseEstimator):
        pass

    def inverse_transform(self, X, y=None, **kwargs):
        if hasattr(self.base_element, "inverse_transform"):
            # todo: check this
            X, y, kwargs = self.adjusted_delegate_call(
                self.base_element.inverse_transform, X, y, **kwargs
            )
        return X, y, kwargs

    @property
    def _estimator_type(self):
        estimator_types = list()
        for element in self.elements:
            estimator_types.append(getattr(element, "_estimator_type"))

        unique_types = set(estimator_types)
        if len(unique_types) > 1:
            raise NotImplementedError(
                "Switch should only contain elements of a single type (transformer, classifier, "
                "regressor). Found multiple types: {}".format(unique_types)
            )
        elif len(unique_types) == 1:
            return list(unique_types)[0]
        else:
            return

    @property
    def feature_importances_(self):
        if hasattr(self.base_element, "feature_importances_"):
            return getattr(self.base_element, "feature_importances_")


class DataFilter(BaseEstimator, PhotonNative):
    """
    Helper Class to split the data e.g. for stacking.
    """

    def __init__(self, indices):
        self.name = "DataFilter"
        self.hyperparameters = {}
        self.indices = indices
        self.needs_covariates = False
        self.needs_y = False

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Returns only part of the data, column-wise filtered by self.indices
        """
        return X[:, self.indices], y, kwargs

    def copy_me(self):
        return self.__class__(indices=self.indices)

    @property
    def _estimator_type(self):
        return


class CallbackElement(PhotonNative):
    def __init__(self, name, delegate_function, method_to_monitor="transform"):

        self.needs_covariates = True
        self.needs_y = True
        self.name = name
        # todo: check if delegate function accepts X, y, kwargs
        self.delegate_function = delegate_function
        self.method_to_monitor = method_to_monitor
        self.hyperparameters = {}
        self.is_transformer = True
        self.is_estimator = False

    def fit(self, X, y=None, **kwargs):
        if self.method_to_monitor == "fit":
            self.delegate_function(X, y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        if self.method_to_monitor == "transform":
            self.delegate_function(X, y, **kwargs)
        return X, y, kwargs

    def copy_me(self):
        return self.__class__(self.name, self.delegate_function, self.method_to_monitor)

    def inverse_transform(self, X, y=None, **kwargs):
        return X, y, kwargs

    @property
    def _estimator_type(self):
        return None

    @property
    def feature_importances_(self):
        return
