import os.path
import urllib.request
from zipfile import ZipFile
from io import BytesIO, TextIOWrapper
from sklearn.preprocessing import LabelEncoder
import pandas as pd

datasets = [
    {
        'name': 'Abalone',
        'uci_url': 'https://archive.ics.uci.edu/static/public/1/abalone.zip',
        'website': 'https://archive.ics.uci.edu/dataset/1/abalone',
        'target': 'Rings',
        'columns': ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight',
                    'VisceraWeight', 'ShellWeight', 'Rings'],
        'mask_targets': False,
        'categorials': ['Sex'],
        'filename': 'abalone.data'
    },
    {
        'name': "Haberman's Survival",
        'uci_url': 'https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip',
        'website': 'https://archive.ics.uci.edu/dataset/43/haberman+s+survival',
        'columns': ["Age", "Year", "nPositiveNodes", "Survival"],
        'target': 'Survival',
        'categorials': [],
        'filename': 'haberman.data'
    }
]


def load_dataset(name: str = None):
    """
    Load the UCI datasets
    If no name is specified, this function can be used to iterate over X, y.

    Parameters:
        name: str, default=None
            Name of the dataset to load
    """
    if name is not None:
        target_dataset = None
        for ds in datasets:
            if ds['name'] == name:
                target_dataset = ds
                break
        if target_dataset is None:
            raise ValueError("Requested dataset not found")
        df = __load_single_dataset(target_dataset)
        return df.loc[:, df.columns != target_dataset['target']].to_numpy(), df[target_dataset['target']].to_numpy()
    for dataset in datasets:
        df = __load_single_dataset(dataset)
        yield df.loc[:, df.columns != dataset['target']].to_numpy(), df[dataset['target']].to_numpy()


def __load_single_dataset(dataset: dict):
    if os.path.isfile(dataset['filename']):
        print(f"Using cached dataset {dataset['filename']}")
        return pd.read_csv(dataset['filename'])
    print(f"Fetching dataset {dataset['filename']}")
    resp = urllib.request.urlopen(dataset['uci_url'])
    zipfile = ZipFile(BytesIO(resp.read()))
    in_mem_fo = TextIOWrapper(zipfile.open(dataset['filename']), encoding='utf-8')
    df = pd.read_csv(in_mem_fo, header=None, names=dataset['columns'])
    # remove categorials
    for categorial in dataset['categorials']:
        df[categorial] = LabelEncoder().fit_transform(df[categorial])
    df.to_csv(dataset['filename'], index=False)
    return df


if __name__ == '__main__':
    for X, y in load_dataset():
        print(f'{X.shape}, {y.shape}')
