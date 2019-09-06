import inspect
import json
import os


class ElementDictionary:
    PHOTON_REGISTRIES = ['PhotonCore', 'PhotonNeuro']

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

        folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        file_name = os.path.join(folder, photon_package + '.json')
        if os.path.isfile(file_name):
            # Reading json
            with open(file_name, 'r') as f:
                file_content = json.load(f)
        else:
            file_content = dict()
            with open(file_name, 'w') as f:
                json.dump(file_content, f)

        return file_content, file_name

    @staticmethod
    def get_package_info(photon_package: list = PHOTON_REGISTRIES) -> dict:
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

            content, _ = ElementDictionary.get_json(package)

            for key in content:
                class_path, class_name = os.path.splitext(content[key][0])
                class_info[key] = class_path, class_name[1:]
        return class_info
