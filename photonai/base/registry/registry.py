import inspect
import json
import os
import sys
from os import path

from photonai.photonlogger import Logger
from photonai.base.photon_elements import PipelineElement


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

    Example
    -------
        from photonai.configuration.Register import PhotonRegister

        # get info about object, name, namespace and possible hyperparameters
        PhotonRegister.info("SVC")

        # show all items that are registered
        PhotonRegister.list()

        # register new object
        PhotonRegister.save("ABC1", "namespace.filename.ABC1", "Transformer")

        # delete it again.
        PhotonRegister.delete("ABC1")

    """

    PHOTON_REGISTRIES = ['PhotonCore', 'PhotonNeuro']

    def __init__(self):
        self.custom_elements = None
        self.custom_elements_folder = None
        self.custom_elements_file = None

    def _load_custom_folder(self, custom_elements_folder):
        self.custom_elements_folder, self.custom_elements_file = self._check_custom_folder(custom_elements_folder)

        # load custom elements from json
        self.custom_elements = self._load_json('CustomElements')
        self.PHOTON_REGISTRIES.append('CustomElements')

    @staticmethod
    def _check_custom_folder(custom_folder):
        if not path.exists(custom_folder):
            Logger().info('Creating folder {}'.format(custom_folder))
            os.makedirs(custom_folder)

        custom_file = path.join(custom_folder, 'CustomElements.json')
        if not path.isfile(custom_file):
            Logger().info('Creating CustomElements.json')
            with open(custom_file, 'w') as f:
                json.dump('', f)

        return custom_folder, custom_file

    def activate(self, custom_elements_folder):
        if not path.exists(custom_elements_folder):
            raise FileNotFoundError("Couldn't find custom elements folder: {}".format(custom_elements_folder))
        if not path.isfile(path.join(custom_elements_folder, 'CustomElements.json')):
            raise FileNotFoundError("Couldn't find CustomElements.json. Did you register your element first?")

        self._load_custom_folder(custom_elements_folder)

        # add folder to python path
        Logger().info("Adding custom elements folder to system path...")
        sys.path.append(self.custom_elements_folder)

        PipelineElement.ELEMENT_DICTIONARY.update(self._get_package_info(['CustomElements']))
        Logger().info('Successfully activated custom elements!')

    def register(self, photon_name: str, class_str: str, element_type: str, custom_folder: str):
        """
        Save element information to the JSON file

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal with which you want to access the class
        * 'class_str' [str]:
          The namespace of the class, like in the import statement
        * 'element_type' [str]:
          Can be 'Estimator' or 'Transformer'
        * 'custom_folder' [str]:
          All registrations are saved to this folder
        """

        # check if folder exists
        self._load_custom_folder(custom_folder)

        if not element_type == "Estimator" and not element_type == "Transformer":
            Logger().error("Variable element_type must be 'Estimator' or 'Transformer'")

        duplicate = self._check_duplicate(photon_name=photon_name, class_str=class_str, content=self.custom_elements)

        if not duplicate:
            python_file = path.join(self.custom_elements_folder, class_str.split('.')[0] + '.py')
            if not path.isfile(python_file):
                raise FileNotFoundError("Couldn't find python file {} in your custom elements folder. "
                                        "Please copy your file into this folder first!".format(python_file))
            # add new element
            self.custom_elements[photon_name] = class_str, element_type

            # write back to file
            self._write2json(self.custom_elements)
            Logger().info('Adding PipelineElement ' + class_str + ' to CustomElements.json as "' + photon_name + '".')
        else:
            Logger().error('Could not register element!')

    def info(self, photon_name):
        """
        Show information for object that is encoded by this name.

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal which accesses the class
        """
        content = self._get_package_info()  # load existing json

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
                Logger().error(e)
                Logger().error("Could not instantiate class " + element_namespace + "." + element_name)
        else:
            Logger().error("Could not find element " + photon_name)

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
        Logger().info('Removing the PipelineElement named "{0}" from CustomElements.json.'.format(photon_name))

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
            Logger().info('A PipelineElement named ' + photon_name + ' has already been registered.')
            return True

        # check for duplicate class_str
        if any(class_str in '.'.join([s[0], s[1]]) for s in content.values()):
            Logger().info('The Class named ' + class_str + ' has already been registered.')
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

        if photon_package == 'CustomElements':
            folder = self.custom_elements_folder
        else:
            folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        file_name = path.join(folder, photon_package + '.json')
        file_content = {}

        # Reading json
        with open(file_name, 'r') as f:
            try:
                file_content = json.load(f)
            except json.JSONDecodeError as jde:
                # handle empty file
                if jde.msg == 'Expecting value':
                    Logger().error("Package File " + file_name + " was empty.")
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
        with open(self.custom_elements_file, 'w') as f:
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

    def list_available_elements(self, photon_package: list = PHOTON_REGISTRIES):
        """
        Print info about all items that are registered for the PHOTON submodule to the console.

        Parameters:
        -----------
        * 'photon_package' [list]:
          The names of the PHOTON submodules for which the elements should be retrieved
        """
        for package in photon_package:
            content = self._load_json(package)
            if len(content) > 0:
                print('\n' + package)
                for k, v in sorted(content.items()):
                    class_info, package_type = v
                    print("{:<35} {:<75} {:<5}".format(k, class_info, package_type))
