import inspect
import importlib
import json
import os
import sys
from glob import glob

import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from shutil import copyfile

from photonai.photonlogger.logger import logger


class PhotonRegistry:
    """
    Helper class to manage the PHOTONAI Element Register.

    Use it to add and remove items into the register.
    You can also retrieve information about items and its hyperparameters.

    Every item in the register is encoded by a string literal
    that points to a python class and its namespace.
    You can access the python class via the string literal.
    The class PhotonElement imports and instantiates the class for you.

    Example:
        ``` python
        import os
        from photonai.base import PhotonRegistry

        # REGISTER ELEMENT saved in folder custom_elements_folder
        base_folder = os.path.dirname(os.path.abspath(__file__))
        custom_elements_folder = os.path.join(base_folder, 'custom_elements')

        registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)
        registry.register(photon_name='MyCustomEstimator',
                          class_str='custom_estimator.CustomEstimator',
                          element_type='Estimator')

        registry.activate()
        registry.info('MyCustomEstimator')

        # get informations of other available elements
        registry.info('SVC')
        ```

    """

    ELEMENT_DICTIONARY = dict()
    PHOTON_REGISTRIES = ['PhotonCore']
    CUSTOM_ELEMENTS_FOLDER = None
    CUSTOM_ELEMENTS = None

    def __init__(self, custom_elements_folder: str = None):
        """
        Initialize the object.

        Parameters:
            custom_elements_folder:
                Path to folder with custom element in it.

        """
        self.current_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.module_path = os.path.join(self.current_folder, "modules")
        if not os.path.isdir(self.module_path):
            os.mkdir(self.module_path)

        # update list with available sub_elements
        self._list_available_modules()
        PhotonRegistry.CUSTOM_ELEMENTS_FOLDER = custom_elements_folder
        self._load_custom_folder(custom_elements_folder)

        if len(PhotonRegistry.ELEMENT_DICTIONARY) == 0 or \
                PhotonRegistry.ELEMENT_DICTIONARY == PhotonRegistry.CUSTOM_ELEMENTS:
            PhotonRegistry.ELEMENT_DICTIONARY.update(self.get_package_info())

    def _list_available_modules(self):
        for abs_filename in glob(os.path.join(self.module_path, "*.json")):
            basename = os.path.basename(abs_filename)
            file, ext = os.path.splitext(basename)
            if file not in PhotonRegistry.PHOTON_REGISTRIES:
                PhotonRegistry.PHOTON_REGISTRIES.append(file)

    def add_module(self, path_to_file: str):
        filename = os.path.basename(path_to_file)
        copyfile(path_to_file, os.path.join(self.module_path, filename))
        self._list_available_modules()
        PhotonRegistry.ELEMENT_DICTIONARY = self.get_package_info()

    def delete_module(self, module_name: str):
        PhotonRegistry.PHOTON_REGISTRIES.remove(module_name)
        os.remove(os.path.join(self.module_path, module_name + ".json"))
        PhotonRegistry.ELEMENT_DICTIONARY = self.get_package_info()

    def _load_json(self, photon_package: str) -> dict:
        """
        Load JSON file in which the elements for the PHOTON submodule are stored.

        The JSON files are stored in the framework folder by the name convention 'photon_package.json'

        Parameters:
            photon_package:
              The name of the photonai submodule

        Returns:
            JSON file as dict, file path as str.

        """
        if photon_package == 'CustomElements':
            folder = PhotonRegistry.CUSTOM_ELEMENTS_FOLDER
            if not folder:
                return {}
        elif photon_package == "PhotonCore":
            folder = self.current_folder
        else:
            folder = self.module_path

        file_name = os.path.join(folder, photon_package + '.json')
        file_content = {}

        # Reading json
        try:
            with open(file_name, 'r') as f:
                try:
                    file_content = json.load(f)
                except json.JSONDecodeError as jde:
                    # handle empty file
                    if jde.msg == 'Expecting value':
                        logger.error("Package File " + file_name + " was empty.")
                    else:
                        raise jde
        except FileNotFoundError:
            logger.error("Could not find file for " + photon_package)
        if not file_content:
            file_content = dict()
        return file_content

    def get_package_info(self, photon_package: list = PHOTON_REGISTRIES) -> dict:
        """
        Collect all registered elements from JSON file.

        Parameters:
            photon_package:
                The names of the PHOTONAI submodules for which
                the elements should be retrieved.

        Returns:
            Dict of registered elements.

        """
        class_info = dict()
        for package in photon_package:

            content = self._load_json(package)

            for idx, key in enumerate(content):
                class_path, class_name = os.path.splitext(content[key][0])

                if idx == 0 and package not in ["PhotonCore", "CustomElements"]:
                    # try to import something from module.
                    # if that fails. drop this shit.
                    try:
                        imported_module = importlib.import_module(class_path)
                        desired_class = getattr(imported_module, class_name[1:])
                        custom_element = desired_class()
                    except (AttributeError, ModuleNotFoundError) as e:
                        logger.error(e)
                        logger.error("Could not import from package {}. Deleting json.".format(package))
                        self.delete_module(package)

                class_info[key] = class_path, class_name[1:]
        return class_info

    def _load_custom_folder(self, custom_elements_folder):

        if PhotonRegistry.CUSTOM_ELEMENTS_FOLDER is not None:
            PhotonRegistry.CUSTOM_ELEMENTS_FOLDER = self._check_custom_folder(custom_elements_folder)

            # load custom elements from json
            PhotonRegistry.CUSTOM_ELEMENTS = self._load_json('CustomElements')
            PhotonRegistry.PHOTON_REGISTRIES.append('CustomElements')
            PhotonRegistry.ELEMENT_DICTIONARY.update(PhotonRegistry.CUSTOM_ELEMENTS)

    @staticmethod
    def _check_custom_folder(custom_folder):
        if not os.path.exists(custom_folder):
            logger.info('Creating folder {}'.format(custom_folder))
            os.makedirs(custom_folder)

        custom_file = os.path.join(custom_folder, 'CustomElements.json')
        if not os.path.isfile(custom_file):
            logger.info('Creating CustomElements.json')
            with open(custom_file, 'w') as f:
                json.dump('', f)

        return custom_folder

    def activate(self):
        if not PhotonRegistry.CUSTOM_ELEMENTS_FOLDER:
            raise ValueError("To activate a custom elements folder, specify a folder when instantiating the registry "
                             "module. Example: registry = PhotonRegistry('/MY/CUSTOM/ELEMENTS/FOLDER) "
                             "In case you don't have any custom models, there is no need to activate the registry.")
        if not os.path.exists(PhotonRegistry.CUSTOM_ELEMENTS_FOLDER):
            raise FileNotFoundError("Couldn't find custom elements folder: {}".format(PhotonRegistry.CUSTOM_ELEMENTS_FOLDER))
        if not os.path.isfile(os.path.join(PhotonRegistry.CUSTOM_ELEMENTS_FOLDER, 'CustomElements.json')):
            raise FileNotFoundError("Couldn't find CustomElements.json. Did you register your element first?")

        # add folder to python path
        logger.info("Adding custom elements folder to system path...")
        sys.path.append(PhotonRegistry.CUSTOM_ELEMENTS_FOLDER)

        PhotonRegistry.ELEMENT_DICTIONARY.update(self.get_package_info(['CustomElements']))
        logger.info('Successfully activated custom elements!')

    def register(self, photon_name: str, class_str: str, element_type: str):
        """
        Save element information to the JSON file.

        Parameters:
            photon_name:
                The string literal with which you want to access the class.

            class_str:
                The namespace of the class, like in the import statement.

            element_type:
                Can be 'Estimator' or 'Transformer'

        """
        # check if folder exists
        if not PhotonRegistry.CUSTOM_ELEMENTS_FOLDER:
            raise ValueError("To register an element, specify a custom elements folder when instantiating the registry "
                             "module. Example: registry = PhotonRegistry('/MY/CUSTOM/ELEMENTS/FOLDER)")

        if not element_type == "Estimator" and not element_type == "Transformer":
            raise ValueError("Variable element_type must be 'Estimator' or 'Transformer'")

        duplicate = self._check_duplicate(photon_name=photon_name, class_str=class_str, content=PhotonRegistry.CUSTOM_ELEMENTS)

        if not duplicate:
            python_file = os.path.join(PhotonRegistry.CUSTOM_ELEMENTS_FOLDER, class_str.split('.')[0] + '.py')
            if not os.path.isfile(python_file):
                raise FileNotFoundError("Couldn't find python file {} in your custom elements folder. "
                                        "Please copy your file into this folder first!".format(python_file))
            # add new element
            PhotonRegistry.CUSTOM_ELEMENTS[photon_name] = class_str, element_type

            # write back to file
            self._write_to_json(PhotonRegistry.CUSTOM_ELEMENTS)
            logger.info('Adding PipelineElement ' + class_str + ' to CustomElements.json as "' + photon_name + '".')

            # activate custom elements
            self.activate()

            # check custom element
            logger.info("Running tests on custom element...")
            return self._run_tests(photon_name, element_type)
        else:
            logger.error('Could not register element!')

    def _run_tests(self, photon_name, element_type):

        # this is a sneaky hack to avoid circular imports of PipelineElement
        imported_module = importlib.import_module("photonai.base")
        desired_class = getattr(imported_module, "PipelineElement")
        custom_element = desired_class(photon_name)

        # check if has fit, transform, predict
        if not hasattr(custom_element.base_element, 'fit'):
            raise NotImplementedError("Custom element does not implement fit() method.")

        if element_type == 'Transformer' and not hasattr(custom_element.base_element, 'transform'):
            raise NotImplementedError("Custom element does not implement transform() method.")

        if element_type == 'Estimator' and not hasattr(custom_element.base_element, 'predict'):
            raise NotImplementedError("Custom element does not implement predict() method.")

        # check if estimator is regressor or classifier
        if element_type == 'Estimator':
            if hasattr(custom_element, '_estimator_type'):
                est_type = getattr(custom_element, '_estimator_type')
                if est_type == "regressor":
                    X, y = load_boston(return_X_y=True)
                elif est_type == "classifier":
                    X, y = load_breast_cancer(return_X_y=True)
                else:
                    raise ValueError("Custom element does not specify whether it is a regressor or classifier. "
                                     "Is {}".format(est_type))
            else:
                raise NotImplementedError("Custom element does not specify whether it is a regressor or classifier. "
                                          "Consider inheritance from ClassifierMixin or RegressorMixin or set "
                                          "_estimator_type manually.")
        else:
            X, y = load_boston(return_X_y=True)

        # try and test functionality
        kwargs = {'covariates': np.random.randn(len(y))}

        try:
            # test fit
            returned_element = custom_element.base_element.fit(X, y, **kwargs)
        except Exception as e:
            logger.info("Not able to run tests on fit() method. Test data not compatible.")
            return e

        if not isinstance(returned_element, custom_element.base_element.__class__):
            raise NotImplementedError("fit() method does not return self.")

        try:
            # test transform or predict (if base element does not implement transform method, predict should be called
            # by PipelineElement -> so we only need to test transform()
            if custom_element.needs_y:
                if element_type == 'Estimator':
                    raise NotImplementedError("Estimator should not need y.")
                Xt, yt, kwargst = custom_element.base_element.transform(X, y, **kwargs)
                if 'covariates' not in kwargst.keys():
                    raise ValueError("Custom element does not correctly transform kwargs although needs_y is True. "
                                     "If you change the number of samples in transform(), make sure to transform kwargs "
                                     "respectively.")
                if not len(kwargst['covariates']) == len(X):
                    raise ValueError("Custom element is not returning the correct number of samples!")

            elif custom_element.needs_covariates:
                if element_type == 'Estimator':
                    yt, kwargst = custom_element.base_element.predict(X, **kwargs)
                    if not len(yt) == len(y):
                        raise ValueError("Custom element is not returning the correct number of samples!")
                else:
                    Xt, kwargst = custom_element.base_element.transform(X, **kwargs)

                    if not len(Xt) == len(X) or not len(kwargst['covariates']) == len(X):
                        raise ValueError("Custom element is not returning the correct number of samples!")

            else:
                if element_type == 'Estimator':
                    yt = custom_element.base_element.predict(X)
                    if not len(yt) == len(y):
                        raise ValueError("Custom element is not returning the correct number of samples!")
                else:
                    Xt = custom_element.base_element.transform(X)
                    if not len(Xt) == len(X):
                        raise ValueError("Custom element is not returning the correct number of samples!")

        except ValueError as ve:
            if "too many values to unpack" in ve.args[0]:
                raise ValueError("Custom element does not return X, y and kwargs the way it should "
                                          "according to needs_y and needs_covariates.")
            else:
                logger.info(ve.args[0])
                return ve
        except Exception as e:
            logger.info(e.args[0])
            logger.info("Not able to run tests on transform() or predict() method. Test data probably not compatible.")
            return e

        logger.info('All tests on custom element passed.')

    def info(self, photon_name: str):
        """
        Show information for object that is encoded by this name.

        Parameters:
            photon_name:
                The string literal which accesses the class.

        """
        content = self.get_package_info()  # load existing json

        if photon_name in content:
            element_namespace, element_name = content[photon_name]

            print("----------------------------------")
            print("Name: " + element_name)
            print("Namespace: " + element_namespace)
            print("----------------------------------")

            try:
                imported_module = __import__(element_namespace, globals(), locals(), element_name, 0)
                desired_class = getattr(imported_module, element_name)
                base_element = desired_class()
                print("Possible Hyperparameters as derived from constructor:")
                class_args = inspect.signature(base_element.__init__)
                for item, more_info in class_args.parameters.items():
                    print("{:<35} {:<75}".format(item, str(more_info)))
                print("----------------------------------")
            except Exception as e:
                logger.error(e)
                logger.error("Could not instantiate class " + element_namespace + "." + element_name)
        else:
            logger.error("Could not find element " + photon_name)

    def delete(self, photon_name: str):
        """
        Delete Element from JSON file.

        Parameters:
            photon_name:
                The string literal encoding the class.

        """
        if photon_name in PhotonRegistry.CUSTOM_ELEMENTS:
            del PhotonRegistry.CUSTOM_ELEMENTS[photon_name]

            self._write_to_json(PhotonRegistry.CUSTOM_ELEMENTS)
            logger.info('Removing the PipelineElement named "{0}" from CustomElements.json.'.format(photon_name))
        else:
            logger.info('Cannot remove "{0}" from CustomElements.json. Element has not been registered before.'.format(photon_name))

    @staticmethod
    def _check_duplicate(photon_name, class_str: str, content: str) -> bool:
        """
        Helper function to check if the entry is either registered by
        a different name or if the name is already given to another class.

         Parameters:
            content:
                The content of the CustomElements.json.

            class_str:
                The namespace.Classname, where the class lives,
                from where it should be imported.

            photon_name:
                The name of the element with which it is called within PHOTONAI.

        Returns:
             False if there is no key with this name and the class
             is not already registered with another key.

        """
        # check for duplicate name (dict key)
        if photon_name in content:
            logger.info('A PipelineElement named ' + photon_name + ' has already been registered.')
            return True

        # check for duplicate class_str
        if any(class_str in '.'.join([s[0], s[1]]) for s in content.values()):
            logger.info('The Class named ' + class_str + ' has already been registered.')
            return True
        return False

    def _write_to_json(self, content_to_write: dict):
        """
        Write json content to file

        Parameters:
            content2write:
                The new information to attach to the file.

        """
        # Writing JSON data
        with open(os.path.join(self.CUSTOM_ELEMENTS_FOLDER, "CustomElements.json"), 'w') as f:
            json.dump(content_to_write, f)

    def list_available_elements(self, photon_package: list = PHOTON_REGISTRIES):
        """
        Print info about all items that are registered for the PHOTONAI
        submodule to the console.

        Parameters:
            photon_package:
                The names of the PHOTON submodules for which
                the elements should be retrieved.

        """
        if isinstance(photon_package, str):
            photon_package = [photon_package]
        for package in photon_package:
            content = self._load_json(package)
            if len(content) > 0:
                print('\n' + package)
                for k, v in sorted(content.items()):
                    class_info, package_type = v
                    print("{:<35} {:<75} {:<5}".format(k, class_info, package_type))
