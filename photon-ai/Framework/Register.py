# register Elements, Optimizers, ...?
import inspect
import json
import os.path
from logging import Logger


class RegisterPipelineElement:
    """
    Helper class to register new pipeline elements into the element register json file.

    You can use it to add and remove items into the register.
    Every item in the register is encoded by a string literal that points to a python class and its namespace.
    You can access the python class via the string literal.

    Example:
        # Create new entry information
        new_reg_element = RegisterPipelineElement('photon_core', 'svc', 'sklearn.svm.svc', 'estimator')
        # Save it to JSON file
        new_reg_element.save()
        # Remove it from JSON file
        new_reg_element.delete()
    """
    def __init__(self, photon_package: str, photon_name: str, class_str: str, element_type=None):
        """
        Create new entry information
        :param photon_package: the photon-ai module name e.g. photon_core, photon_neuro
        :param photon_name: the string literal with which you want to access the class
        :param class_str: the namespace of the class, like in the import statement
        :param element_type: 'estimator' or 'transformer'
        """
        self.photon_name = photon_name
        self.photon_package = photon_package
        self.class_str = class_str
        self.element_type = element_type

    def save(self):
        """
        Save Element information to the JSON file
        """
        # register_element and jsonify
        content, _ = PhotonRegister.get_json(self.photon_package)  # load existing json
        duplicate = self.check_duplicate(content)
        if not duplicate:
            Logger().info('Adding PipelineElement ' + self.class_str + ' to ' +
                          self.photon_package + ' as "' + self.photon_name + '".')

            # add new element
            content[self.photon_name] = self.class_str, self.element_type

            # write back to file
            PhotonRegister.write2json(content, self.photon_package)

    def delete(self):
        """
        Delete Element from JSON file
        """
        content, _ = PhotonRegister.get_json(self.photon_package)  # load existing json
        Logger().info('Removing the PipelineElement named "{0}" ({1}.{2}) from {3}.'
                      .format(self.photon_name, content[self.photon_name][0], content[self.photon_name][1],
                              self.photon_package))

        if self.photon_name in content:
            del content[self.photon_name]
        PhotonRegister.write2json(content, self.photon_package)

    def check_duplicate(self, content):
        """
        Helper function to check if the entry is either registered by a different name or if the name is already given
        to another class
        :param content: JSON file as dict
        :type content: dict
        :return: bool, False if there is no duplicate
        """
        # check for duplicate name (dict key)
        flag = 0
        if self.photon_name in content:
           flag += 1
           Logger().info('The PipelineElement named "' + self.photon_name + '" has already been registered in '
                + self.photon_package + ' with ' + content[self.photon_name][0] + '.')

        # check for duplicate class_str
        if any(self.class_str in s for s in content.values()):
            flag += 1

            Logger().info('The PipelineElement with the ClassName "' + self.class_str +
                          '" has already been registered in ' + self.photon_package + '". "' +
                          self.photon_name + '" not added to ' + self.photon_package + '.')

        return flag > 0


class PhotonRegister:
    """
    Manages the JSON files for each PHOTON submodule
    """
    def __init__(self):
        pass

    # one json file per Photon Package (Core, Neuro, Genetics, Designer (if necessary)
    @staticmethod
    def get_json(photon_package: str):
        """
        Load JSON file in which the elements for the PHOTON submodule are stored.

        The JSON files are stored in the framework folder by the name convention 'photon_package.json'
        :param photon_package: the name of the photon-ai submodule
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
    def get_package_info(photon_package: str) -> dict:
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
    def show_package_info(photon_package: str):
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


# if __name__ == '__main__':
#     ELEMENT_DICTIONARY = PhotonRegister.get_package_info(['PhotonCore'])
#     PhotonRegister.show_package_info(['PhotonCore'])
#
#

# import sklearn
# import inspect
# for name, obj in inspect.getmembers(sklearn):
#     #print(obj)
#     print(name)

# print(hasattr(PCA(), '_estimator_type'))

# ELEMENT_DICTIONARY = RegisterPipelineElement.get_pipeline_element_infos(['PhotonCore', 'PhotonCore2'])
# print(ELEMENT_DICTIONARY)

# photon_package = 'PhotonCore'  # where to add the element
# photon_name = 'skjhvr'  # element name
# class_str = 'sklearn.svm.SljkVR'  # element info
# element_type = 'Estimator'  # element type
# RegisterPipelineElement(photon_name=photon_name,
#                         photon_package=photon_package,
#                         class_str=class_str,
#                         element_type=element_type).add()
#
# photon_name = 'slkjvr' # elment name
# class_str = 'sklearn.test'  # element info
# RegisterPipelineElement(photon_name=photon_name,
#                         photon_package=photon_package,
#                         class_str=class_str,
#                         element_type=element_type).add()
#
# photon_name = 'Test'  # element name
# class_str = 'sklearn.svm.SVR'  # element info
# RegisterPipelineElement(photon_name=photon_name,
#                         photon_package=photon_package,
#                         class_str=class_str,
#                         element_type=element_type).add()
#
# photon_name = 'PCA'  # element name
# class_str = 'sklearn.decomposition.PCA' # element info
# element_type = 'Transformer' # element type
# RegisterPipelineElement(photon_name=photon_name,
#                         photon_package=photon_package,
#                         class_str=class_str,
#                         element_type=element_type).add()
#
# eldict = RegisterPipelineElement.get_pipeline_element_infos(['PhotonCore'])
# # eldict = PhotonRegister.get_element_infos('PhotonCore')
# print(eldict)


# RegisterPipelineElement(photon_name='PCA',
#                         photon_package=photon_package).remove()


# Write complete dict to json (OVERWRITES EVERYTHING IN IT!!!)

# ELEMENT_DICTIONARY = {'pca': ('sklearn.decomposition.PCA', 'Transformer'),
#                       'svc': ('sklearn.svm.SVC', 'Estimator'),
#                       'knn': ('sklearn.neighbors.KNeighborsClassifier', 'Estimator'),
#                       'logistic': ('sklearn.linear_model.LogisticRegression', 'Estimator'),
#                       'dnn': ('PipelineWrapper.TFDNNClassifier.TFDNNClassifier', 'Estimator'),
#                       'KerasDNNClassifier': ('PipelineWrapper.KerasDNNClassifier.KerasDNNClassifier', 'Estimator'),
#                       'standard_scaler': ('sklearn.preprocessing.StandardScaler', 'Transformer'),
#                       'wrapper_model': ('PipelineWrapper.WrapperModel.WrapperModel', 'Estimator'),
#                       'test_wrapper': ('PipelineWrapper.TestWrapper.WrapperTestElement', 'Estimator'),
#                       'ae_pca': ('PipelineWrapper.PCA_AE_Wrapper.PCA_AE_Wrapper', 'Transformer'),
#                       'rl_cnn': ('photon_core.PipelineWrapper.RLCNN.RLCNN', 'Estimator'),
#                       'CNN1d': ('PipelineWrapper.CNN1d.CNN1d', 'Estimator'),
#                       'SourceSplitter': ('PipelineWrapper.SourceSplitter.SourceSplitter', 'Transformer'),
#                       'f_regression_select_percentile': ('PipelineWrapper.FeatureSelection.FRegressionSelectPercentile', 'Transformer'),
#                       'f_classif_select_percentile': ('PipelineWrapper.FeatureSelection.FClassifSelectPercentile', 'Transformer'),
#                       'py_esn_r': ('PipelineWrapper.PyESNWrapper.PyESNRegressor', 'Estimator'),
#                       'py_esn_c': ('PipelineWrapper.PyESNWrapper.PyESNClassifier', 'Estimator'),
#                       'SVR': ('sklearn.svm.SVR', 'Estimator'),
#                       'KNeighborsRegressor': ('sklearn.neighbors.KNeighborsRegressor', 'Estimator'),
#                       'DecisionTreeRegressor': ('sklearn.tree.DecisionTreeRegressor', 'Estimator'),
#                       'RandomForestRegressor': ('sklearn.ensemble.RandomForestRegressor', 'Estimator'),
#                       'KerasDNNRegressor': ('PipelineWrapper.KerasDNNRegressor.KerasDNNRegressor', 'Estimator'),
#                       'PretrainedCNNClassifier': ('PipelineWrapper.PretrainedCNNClassifier.PretrainedCNNClassifier', 'Estimator')
#                       }
# PhotonRegister.write2json(ELEMENT_DICTIONARY, 'PhotonCore')
