# register Elements, Optimizers, ...?
import inspect
import json
import os.path
from ..photonlogger.Logger import Logger


class PhotonRegister:
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

    PHOTON_REGISTERS = ['PhotonCore', 'PhotonNeuro']

    def __init__(self):
        pass

    @staticmethod
    def save(photon_name: str, class_str: str, element_type: str, photon_package: str = "CustomElements"):
        """
        Save Element information to the JSON file

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal with which you want to access the class
        * 'class_str' [str]:
          The namespace of the class, like in the import statement
        * 'element_type' [str]:
          Can be 'Estimator' or 'Transformer'
        * 'photon_package' [str]:
          The photonai module name e.g. photon_core, photon_neuro

        """
        # register_element and jsonify

        if element_type != "Estimator" or element_type != "Transformer":
            Logger().error("Variable element_type must be 'Estimator' or 'Transformer'")

        duplicate = PhotonRegister.check_duplicate(photon_name, class_str)
        if not duplicate:
            content, _ = PhotonRegister.get_json(photon_package)  # load existing json
            # add new element
            content[photon_name] = class_str, element_type

            # write back to file
            PhotonRegister.write2json(content, photon_package)

            Logger().info('Adding PipelineElement ' + class_str + ' to ' +
                          photon_package + ' as "' + photon_name + '".')


        else:
            Logger().error('Could not register pipeline element due to duplicates.')

    @staticmethod
    def info(photon_name):
        """
        Show information for object that is encoded by this name.

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal which accesses the class
        """
        content = PhotonRegister.get_package_info(PhotonRegister.PHOTON_REGISTERS)  # load existing json

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



    @staticmethod
    def delete(photon_name, photon_package="CustomElements"):
        """
        Delete Element from JSON file

        Parameters:
        -----------
        * 'photon_name' [str]:
          The string literal encoding the class
        """
        content, _ = PhotonRegister.get_json(photon_package)  # load existing json

        if photon_name in content:
            del content[photon_name]

        PhotonRegister.write2json(content, photon_package)
        Logger().info('Removing the PipelineElement named "{0}" from {1}.'
                      .format(photon_name, photon_package))

    @staticmethod
    def check_duplicate(photon_name, class_str):
        """
        Helper function to check if the entry is either registered by a different name or if the name is already given
        to another class

         Parameters:
        -----------
        * 'photon_name' [str]:
          The name with which the class should be accessed
        * 'class_str' [str]:
          The namespace.Classname, where the class lives, from where it should be imported.

        Returns:
        --------
        Bool, False if there is no key with this name and the class is not already registered with another key
        """

        content = PhotonRegister.get_package_info(PhotonRegister.PHOTON_REGISTERS)  # load existing json

        # check for duplicate name (dict key)
        flag = 0
        if photon_name in content:
           flag += 1
           Logger().info('A PipelineElement named ' + photon_name + ' has already been registered.')

        # check for duplicate class_str
        if any(class_str in '.'.join([s[0], s[1]]) for s in content.values()):
            flag += 1
            Logger().info('The Class named ' + class_str + ' has already been registered.')

        return flag > 0

    # one json file per Photon Package (Core, Neuro, Genetics, Designer (if necessary)
    @staticmethod
    def get_json(photon_package: str):
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
        file_name = os.path.dirname(inspect.getfile(PhotonRegister)) + '/' + photon_package + '.json'
        file_content = {}
        if os.path.isfile(file_name):
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
            file_path = file_name
        else:
            file_content = dict()
            file_path = None
            print(file_name + ' not found. Creating file.')

        return file_content, file_path

    @staticmethod
    def write2json(content2write: dict, photon_package: str):
        """
        Write json content to file

        Parameters:
        -----------
        * 'content2write' [dict]:
          The new information to attach to the file
        * 'photon_package' [str]:
          The PHOTON submodule name to which the new class belongs, so it is written to the correct json file
        """
        file_name = os.path.dirname(inspect.getfile(PhotonRegister)) + '/' + photon_package + '.json'
        # Writing JSON data
        with open(file_name, 'w') as f:
            json.dump(content2write, f)

    @staticmethod
    def get_package_info(photon_package: list = PHOTON_REGISTERS) -> dict:
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

            content, _ = PhotonRegister.get_json(package)

            for key in content:
                class_path, class_name = os.path.splitext(content[key][0])
                class_info[key] = class_path, class_name[1:]
        return class_info

    @staticmethod
    def list(photon_package: list = PHOTON_REGISTERS):
        """
        Print info about all items that are registered for the PHOTON submodule to the console.

        Parameters:
        -----------
        * 'photon_package' [list]:
          The names of the PHOTON submodules for which the elements should be retrieved
        """
        for package in photon_package:
            content, file_name = PhotonRegister.get_json(package)
            if len(content) > 0 and file_name is not None:
                print('\n' + package + ' (' + file_name + ')')
                for k, v in sorted(content.items()):
                    class_info, package_type = v
                    print("{:<35} {:<75} {:<5}".format(k, class_info, package_type))


if __name__ == "__main__":


    # get info about object
    PhotonRegister.info("SVC")

    # show all items
    PhotonRegister.list()

    # register object again -> should fail
    PhotonRegister.save("SVC", "sklearn.svm.SVC", "Estimator")
    PhotonRegister.save("SVC2", "sklearn.svm.SVC", "Estimator")

    # register new object
    PhotonRegister.save("ABC1", "namespace.filename.ABC1", "Transformer")

    # delete it again.
    PhotonRegister.delete("ABC1")
