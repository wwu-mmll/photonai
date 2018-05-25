"""
Loading Module
"""

# Copyright (c) PHOTON Development Team

import os

import numpy as np
import pandas as pd

from DataLoading import DataLoader as dl


class DataContainer:
    def __init__(self):

        self._data_dict = {}
        self.covariates = {}

    def __iadd__(self, new_data):
        if not issubclass(type(new_data), BaseDataObject):
            raise TypeError("Only Type BaseObject allowed")

        if isinstance(new_data, Covariates):
            # Todo: key errors?
            self.covariates[new_data.name] = new_data
        else:
            name = new_data.__class__.__name__
            # horizontal concat is deprecated
            # if name in self._data_dict:
            #     try:
            #         self._data_dict[name].data = pd.concat([self._data_dict[name].data, new_data.data], axis=1)
            #     except Exception as e:
            #         # Todo: proper error management
            #         print('concatenation not successful', e)
            # else:
            #     self._data_dict[name] = new_data
            self._data_dict[name] = new_data
        return self

    @property
    def features(self):
        if 'Features' in self._data_dict:
            return self._data_dict['Features']
        else:
            return None

    @features.setter
    def features(self, value):
        self._data_dict['Features'] = value

    @property
    def targets(self):
        if 'Targets' in self._data_dict:
            return self._data_dict['Targets']
        else:
            return None

    @targets.setter
    def targets(self, value):
        self._data_dict['Targets'] = value


    # def randomize(self):
    #     dataset = []
    #     labels = []
    #     permutation = np.random.permutation(labels.shape[0])
    #     shuffled_dataset = dataset[permutation, :]
    #     shuffled_labels = labels[permutation]
    #     return shuffled_dataset, shuffled_labels

    def __str__(self):
        self_descriptions = []
        # Todo: prettify covariate printing
        data_objects = [self.features, self.targets, self.covariates]
        for item in data_objects:
            item_name = '\n' + type(item).__name__ + '\n-------------------- \n'
            self_descriptions.append(' '.join([item_name, str(item)]))
        self_descriptions.append('\n')
        return '\n'.join(self_descriptions)


class BaseDataObject:
    def __init__(self, file_or_array, **kwargs):

        self.data = None
        try:
            # if its a file path string instantiate the respective loader
            if isinstance(file_or_array, (str, list)):
                if isinstance(file_or_array, list):
                    filename, file_extension = os.path.splitext(
                        file_or_array[0])
                else:
                    filename, file_extension = os.path.splitext(
                        file_or_array)
                # module = __import__('DataLoading.DataLoading')
                # make first letter uppercase and remove the dot
                class_name = file_extension.title()[1:] + "Loader"
                if not hasattr(dl, class_name):
                    raise TypeError("Cannot load files of type " + file_extension)
                class_ = getattr(dl, class_name)
                instance = class_()
                self.data = instance(file_or_array, **kwargs)
                # replace nans
                # Todo: na_value?
                # Todo: .gz files
                self.data = self.data.fillna(0)

            elif isinstance(file_or_array, np.ndarray):
                # if numpy array, simply add to collection
                self.data = file_or_array
            else:
                raise TypeError("Input must be string, list of "
                                "strings or numpy array.")
        except FileNotFoundError as fnfe:
            print("Sorry could not find file ", file_or_array, fnfe)
            raise fnfe
        except TypeError as te:
            print("Sorry currently this format is not supported.", te)
            raise te
        except AttributeError as ae:
            print("Too many arguments. Remember only Covariates have names.", ae)
        except Exception as unknown:
            print("Unexpected exception while loading data:", unknown)

    @property
    def values(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.values
        else:
            return self.data

    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            data_descriptions = []
            for key in self.data:
                data_descriptions.append(str(self.data[key].describe()))
            return '\n'.join(data_descriptions)
        else:
            return str(self.data)

    def summary(self):
        """Get variables of Data Container"""
        print(self)
        # print(type(self), ':\n')
        # for key in self.data:
        #     tmp_val = self.data[key].describe()
        #     print(tmp_val)
        #     print('\n')


class Features(BaseDataObject):
    def __init__(self, file_or_array, **kwargs):
        BaseDataObject.__init__(self, file_or_array, **kwargs)


class Targets(BaseDataObject):
    def __init__(self, file_or_array, **kwargs):
        BaseDataObject.__init__(self, file_or_array, **kwargs)

    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


class Covariates(BaseDataObject):
    def __init__(self, name, file_or_array, **kwargs):
        BaseDataObject.__init__(self, file_or_array, **kwargs)
        self.name = name

    def __str__(self):
        return ' '.join([self.name, ':', BaseDataObject.__str__(self)])

if __name__ == "__main__":
    ### proper handling
    dc = DataContainer()

    # add features (CSV file)
    dc += Features('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/CorticalMeasuresENIGMA_SurfAvg.csv',
                   na_values='NA')

    # add targets (Xlsx file)
    dc += Targets('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/testExcelFile.xlsx')

    # add covariate (numpy array)
    variable = np.array([1, 2, 3, 4, 5])
    dc += Covariates('age', variable)

    # print all what is inside
    print(dc)

    ### Load Nifti files
    files = ['/home/nils/data/test_nii'
              '/s8m0wrp1F0079_t1mprsagp2iso20150114PsychMACS0079A1s003a1001.nii',
              '/home/nils/data/test_nii/s8m0wrp1F0079_t1mprsagp2iso20150114PsychMACS0079A1s003a1001.nii']
    print('Test list of nifti files:...')
    dc1 = DataContainer()
    dc1 += Features(files)
    print(dc1.features.values.shape)

    print('Test list with vectorization:...')
    dc2 = DataContainer()
    dc2 += Features(files, vectorize=True)
    print(dc2.features.values.shape)

    # error handling
    # --------------------------
    dc3 = DataContainer()
    a_string = '/home/nils/data/test_nii/s8m0wrp1F0079_t1mprsagp2iso20150114PsychMACS0079A1s003a1001.nii'
    a_float = 3.0
    a_dict = {}

    # File Not Found Error
    dc3 += Features("test.mat")
    # Format Not Supported Error
    dc3 += Targets("fail.txt")
    dc3 += Features(a_string)
    dc3 += Features(a_float)
    dc3 += Features(a_dict)

