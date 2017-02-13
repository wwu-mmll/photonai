"""
Module to load .mat-files

I found this code in some forum but don't remember where...
It handles loading mat-files with complicated structures.
"""

import scipy.io as spio


def loadmat(self, filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False,
                        squeeze_me=True)
    return self._check_keys(data)


def _check_keys(self, dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key],
                      spio.matlab.mio5_params.mat_struct):
            dict[key] = self._todict(dict[key])
    return dict


def _todict(self, matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = self._todict(elem)
        else:
            dict[strg] = elem
    return dict