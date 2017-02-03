"""
Loading Module
"""

# Copyright (c) PHOTON Development Team

import numpy as np


class Container(dict):
    """Container object for data

    Creates dictionary-like object that stores all input data.

    """

    features = {}
    targets = {}
    covariates = {}



class DataFrame:
    """Container object for data

    Creates dictionary-like object that stores all input data.

    """
    def __init__(self):
        """Return a data frame object with features, targets and
        covariates"""

        self.features = {}
        self.targets = {}
        self.covariates = {}


    def add(self, data, name, use_as='', ):
        """Add data variable to data frame

        Adds one data variable at a time (i.e. a
        covariate, a target variable or a data array).

        Currently only works with numpy arrays.

        Specify how the input data should be used with use_as='',
        i.e. 'features', 'targets', 'covariate'.

        Parameters
        ----------
        data : numpy array
            Numpy array with arbitrary dimensions
        name: string
            String specifying name of variable
        use_as: {'features', 'targets', 'covariate'}
            String specifying how to use input variable

        Returns
        -------
        data : Container object
            Data container as dictionary.
        """
        if use_as == 'features':
            self.features[name] = data
        elif use_as == 'targets':
            self.targets[name] = data
        elif use_as == 'covariate':
            self.covariates[name] = data
        else:
            raise ValueError('Wrong use_as variable. Options: '
                             '"features", "targets", "covariate".')


    def summary(self):
        """Get variables of Data Frame"""
        print('Features:\n')
        for key in self.features:
            print(key, 'with shape',
                  self.features[key].shape)
            print('Mean', np.mean(self.features[key]))
        print('\n\nTargets:\n')
        for key in self.targets:
            print(key, 'with shape',
                  self.targets[key].shape)
            print('Mean', np.mean(self.targets[key]))
        print('\n\nCovariates:\n')
        for key in self.covariates:
            print(key, 'with shape',
                  self.covariates[key].shape)
            print('Mean', np.mean(self.covariates[key]))














