import matlab_io as mio
import pandas as pd

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