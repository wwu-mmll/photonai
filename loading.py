"""
Loading Module
"""

# Copyright (c) PHOTON Development Team

import numpy as np
import pandas as pd

class DataContainer:

    def __init__(self):
        self.features = FeaturesObject()
        self.targets = TargetsObject()
        self.covariates = CovariatesObject()

    def add(self, file, input_type, name, use_as=''):
        dataObject = self.select_use(use_as)

        if input_type == 'numpy':
            dataObject.add_numpy(file, name)

        elif input_type == 'excel':
            dataObject.load_excel(file)

        elif input_type == 'csv':
            dataObject.load_csv(file)

        elif input_type == 'mat':
            pass

        elif input_type == 'hdf5':
            pass

        else:
            raise ValueError('Wrong input_type variable. Options: '
                             'None, "excel", "mat", "hdf5".')


    def select_use(self, use_as):
        if use_as.lower() == 'features':
            return self.features
        elif use_as.lower() == 'targets':
            return self.targets
        elif use_as.lower() == 'covariate':
            return self.covariates
        else:
            raise ValueError('Wrong use_as variable. Options: '
                             '"features", "targets", "covariate".')

    def summary(self):
        self.features.summary()
        self.targets.summary()
        self.covariates.summary()

### implement these as singleton patterns

class BaseObject:

    def __init__(self):
        # Create data dictionary to store the pandas arrays
        # Maybe change this so one doesn't need a dictionary
        # just one variable would be enough, basically...
        # Maybe just a list?
        self.data = {}
        self.class_name = None


    def add_numpy(self, data_in, name):
        # add numpy array to data dictionary
        # check if a pandas dataframe already exists
        self.data[name] = pd.DataFrame({name: data_in})


    def load_excel(self, file, **options):
        # Use pandas to read excel data

        # TO DO
        # pass all options for pandas read_excel functions

        data_in = pd.read_excel(file)

        # For more than one variable, pandas will create a dataframe
        # Split dataframe and save each variable separately
        for key in list(data_in):
            self.data[key] = pd.DataFrame(data_in[key])

    def load_csv(self, file, **options):
        # Use pandas to read excel data

        # TO DO
        # pass all options for pandas read_excel functions

        data_in = pd.read_csv(file)

        # For more than one variable, pandas will create a dataframe
        # Split dataframe and save each variable separately
        for key in list(data_in):
            self.data[key] = pd.DataFrame(data_in[key])

    def summary(self):
        """Get variables of Data Container"""
        print(self.class_name, ':\n')
        for key in self.data:
            print(self.data[key].describe())
            print('\n')

    def get_names(self):
        return self.data.keys()


class FeaturesObject(BaseObject):
    def __init__(self):
        BaseObject.__init__(self)
        self.class_name = 'Features'

    def add_numpy(self, data_in, name):
        # add numpy array to data dictionary
        # check if a pandas dataframe already exists
        self.data[name] = pd.DataFrame(data_in)


class TargetsObject(BaseObject):
    def __init__(self):
        BaseObject.__init__(self)
        self.class_name = 'Targets'


class CovariatesObject(BaseObject):
    def __init__(self):
        BaseObject.__init__(self)
        self.class_name = 'Covariates'




