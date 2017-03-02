"""
Loading Module
"""

# Copyright (c) PHOTON Development Team

import os
import numpy as np
import pandas as pd
import DataLoader.DataLoader as dl


class DataContainer:
    def __init__(self):
        self.data_objects = []

    def __iadd__(self, new_data):
        if not issubclass(type(new_data), BaseObject):
            raise TypeError("Only Type BaseObject allowed")
        self.data_objects.append(new_data)
        return self

    def randomize(self):
        dataset = []
        labels = []
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def __str__(self):
        self_descriptions = []
        for item in self.data_objects:
            item_name = '\n' + type(item).__name__ + '\n-------------------- \n'
            self_descriptions.append(' '.join([item_name, str(item)]))
        self_descriptions.append('\n')
        return '\n'.join(self_descriptions)


class BaseObject:
    def __init__(self, file_or_array, **kwargs):

        self.data = None
        try:
            # if its a file path string instantiate the respective loader
            if isinstance(file_or_array, str):
                filename, file_extension = os.path.splitext(file_or_array)
                # module = __import__('DataLoader.DataLoader')
                # make first letter uppercase and remove the dot
                class_name = file_extension.title()[1:] + "Loader"
                if not hasattr(dl, class_name):
                    raise TypeError("Cannot load files of type " + file_extension)
                class_ = getattr(dl, class_name)
                instance = class_()
                self.data = instance(file_or_array, **kwargs)
            else:
                # else simply add to collection
                self.data = file_or_array
        except FileNotFoundError as fnfe:
            print("Sorry could not find file ", file_or_array, fnfe)
        except TypeError as te:
            print("Sorry currently this format is not supported.", te)
        except AttributeError as ae:
            print("Too many arguments. Remember only Covariates have names.", ae)
        except Exception as unknown:
            print("Unexpected exception while loading data:", unknown)

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


class Features(BaseObject):
    def __init__(self, file_or_array, **kwargs):
        BaseObject.__init__(self, file_or_array, **kwargs)


class Targets(BaseObject):
    def __init__(self, file_or_array, **kwargs):
        BaseObject.__init__(self, file_or_array, **kwargs)

    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


class Covariates(BaseObject):
    def __init__(self, name, file_or_array, **kwargs):
        BaseObject.__init__(self, file_or_array, **kwargs)
        self.name = name

    def __str__(self):
        return ' '.join([self.name, ':', BaseObject.__str__(self)])

if __name__ == "__main__":
    # proper handling
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

    # error handling
    # --------------------------
    dc1 = DataContainer()
    # File Not Found Error
    dc1 += Features("test.mat")
    # Format Not Supported Error
    dc1 += Targets("fail.txt")
