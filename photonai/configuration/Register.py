# register Elements, Optimizers, ...?
import inspect
import json
import os.path
from photonai.logging.Logger import Logger


class RegisterPipelineElement:

    def __init__(self, photon_package: str, photon_name: str, class_str: str, element_type=None):
        """
        Create new entry information
        :param photon_package: the photonai module name e.g. photon_core, photon_neuro
        :param photon_name: the string literal with which you want to access the class
        :param class_str: the namespace of the class, like in the import statement
        :param element_type: 'estimator' or 'transformer'
        """




class PhotonRegister:
    """
    Helper class to manage the PHOTON Element Register.
    There is a distinct json file with the elements registered for each photon package (core, neuro, genetics, ..)
    There is also a json file for the user's custom elements.

    You can use it to add and remove items into the register.
    You can also retrieve information about a pipeline element and its hyperparameters.
    Every item in the register is encoded by a string literal that points to a python class and its namespace.
    You can access the python class via the string literal.

    Example:
    --------
    # get info about object
    PhotonRegister.info("SVC")

    # show all items
    PhotonRegister.list()

    # register new object
    PhotonRegister.save("ABC1", "namespace.filename.ABC1", "Transformer")

    # delete it again.
    PhotonRegister.delete("ABC1")


    """

    PHOTON_REGISTERS = ['PhotonCore', 'PhotonNeuro', 'CustomElements']

    def __init__(self):
        pass

    @staticmethod
    def save(photon_name, class_str, element_type, photon_package="CustomElements"):
        """
        Save Element information to the JSON file
        """
        # register_element and jsonify

        if element_type != "Estimator" or "Transformer":
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
        :param photon_name:
        :return:
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
        :param content: JSON file as dict
        :type content: dict
        :return: bool, False if there is no duplicate
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
        :param photon_package: the name of the photonai submodule
        :type photon_package: str
        :return: JSON file as dict, file path as str
        """
        file_name = os.path.dirname(inspect.getfile(PhotonRegister)) + '/' + photon_package + '.json'
        if os.path.isfile(file_name):
            # Reading json
            with open(file_name, 'r') as f:
                file_content = json.load(f)
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
        :param content2write: the new information to attach to the file
        :type content2write: dict
        :param photon_package: the PHOTON submodule name
        :type photon_package: str
        """
        file_name = os.path.dirname(inspect.getfile(PhotonRegister)) + '/' + photon_package + '.json'
        # Writing JSON data
        with open(file_name, 'w') as f:
            json.dump(content2write, f)

    @staticmethod
    def get_package_info(photon_package: list = PHOTON_REGISTERS) -> dict:
        """
        Collect all registered elements from JSON file
        :return: dict of registered elements
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
        :param photon_package: the name of the PHOTON submodule for which the elements should be retrieved
        :type photon_package: str
        """
        for package in photon_package:
            content, file_name = PhotonRegister.get_json(package)
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
