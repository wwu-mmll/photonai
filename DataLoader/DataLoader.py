import pandas as pd
import nibabel as nib
import os
import numpy as np

from DataLoader import MatlabLoader as mio


#Todo: make sure that each class is returning an pandas DataFrame Object


class MatLoader(object):

    def __call__(self, filename, **kwargs):
        mat_data = mio.loadmat(filename)
        if 'var_name' in kwargs:
            var_name = kwargs.get('var_name')
            mat_data = mat_data[var_name]
        return pd.DataFrame(data=mat_data)


class CsvLoader(object):

    def __call__(self, filename, **kwargs):
        csv_data = pd.read_csv(filename, **kwargs)
        return csv_data


class XlsxLoader(object):

    def __call__(self, filename, **kwargs):
        return pd.read_excel(filename)

class NiiLoader(object):
    # Todo: Currently only works when reshaping 3d nii to a 1d vector

    def __call__(self, directory, **kwargs):
        # get nifti filenames in specified directory
        # get rid of asterisk in directory
        # os.path.split will take head and tail of path
        # only use the head
        directory, _ = os.path.split(directory)
        filenames = self.get_filenames(directory)

        # iterate over and load every .nii file
        # this requires one nifti per subject
        data = []
        for ind_sub in range(len(filenames)):
            filename = os.path.join(directory, filenames[ind_sub])
            img = nib.load(filename)
            data.append(img.get_data())

        # stack list elements to matrix
        data = np.stack(data, axis=0)
        data = np.reshape(data, (data.shape[0], data.shape[1] *
                                 data.shape[2] * data.shape[3]))
        # save as numpy array and load with memory map
        np.save((directory + '/photon_data.npy'), data)
        del data
        data = np.load((directory + '/photon_data.npy'), mmap_mode='r')
        return pd.DataFrame(data)


    def get_filenames(self, directory):
        filenames = []
        for file in os.listdir(directory):
            if file.endswith(".nii"):
                filenames.append(file)
        # check if files have been found
        if len(filenames) == 0:
            raise ValueError('There are no .nii-files in the '
                             'specified folder!')
        else:
            return filenames