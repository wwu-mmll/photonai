import os.path
import urllib.request
from zipfile import ZipFile
from io import BytesIO, TextIOWrapper
from sklearn.preprocessing import LabelEncoder
from scipy.io.arff import loadarff
import pandas as pd

datasets = [
    {
        'name': 'Abalone',
        'uci_url': 'https://archive.ics.uci.edu/static/public/1/abalone.zip',
        'website': 'https://archive.ics.uci.edu/dataset/1/abalone',
        'target': 'Rings',
        'columns': ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight',
                    'VisceraWeight', 'ShellWeight', 'Rings'],
        'categorials': ['Sex'],
        'filename': 'abalone.data',
        'task': 'Classification'
    },
    {
        'name': "Haberman's Survival",
        'uci_url': 'https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip',
        'website': 'https://archive.ics.uci.edu/dataset/43/haberman+s+survival',
        'columns': ["Age", "Year", "nPositiveNodes", "Survival"],
        'target': 'Survival',
        'categorials': [],
        'filename': 'haberman.data',
        'task': 'Classification'
    },
    {
        'name': 'Autistic Spectrum Disorder Screening Data for Children',
        'uci_url': 'https://archive.ics.uci.edu/static/public/419/autistic+spectrum+disorder+screening+data+for+children.zip',
        'website': 'https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children',
        'columns': [],
        'target': 'Class/ASD',
        'categorials': ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation', 'Class/ASD'],
        'filename': 'Autism-Child-Data.arff',
        'task': 'Classification'
    },
    {
        'name': 'Parkinsons Telemonitoring Data Set',
        'uci_url': 'https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip',
        'website': 'https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring',
        'columns': ['subject', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'Jitter', 'variation',
                    'shimmer', 'amplitude', 'nhr', 'rpde', 'dfa', 'ppe'],
        'target': 'motor_UPDRS',
        'categorials': [],
        'filename': 'parkinsons_updrs.data',
        'task': 'Regression'
    },
    {

    }
]


def get_available_datasets(print_to_console: bool = False,
                           return_details: bool = False):
    if print_to_console:
        for ds in datasets:
            print('*'*50)
            print(ds['name'])
            if return_details:
                print(f"Details: {ds['website']}")
    if not return_details:
        return [ds['name'] for ds in datasets]
    return [{'name': ds['name'], 'details': ds['website'], 'task': ds['task']} for ds in datasets]



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
        print(f"Using cached dataset {dataset['filename']} ({dataset['task']})")
        return pd.read_csv(dataset['filename'])
    print(f"Fetching dataset {dataset['filename']} ({dataset['task']})")
    resp = urllib.request.urlopen(dataset['uci_url'])
    zipfile = ZipFile(BytesIO(resp.read()))
    in_mem_fo = TextIOWrapper(zipfile.open(dataset['filename']), encoding='utf-8')
    if dataset['filename'][-4:] == 'data':
        df = pd.read_csv(in_mem_fo, header=None, names=dataset['columns'])
    elif dataset['filename'][-4:] == 'arff':
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].str.decode('UTF-8')
    else:
        raise ValueError(f"Unknown filetype: {dataset['filename'][:-4]}")
    # remove categorials
    for categorial in dataset['categorials']:
        df[categorial] = LabelEncoder().fit_transform(df[categorial])
    df.to_csv(dataset['filename'], index=False)
    return df


if __name__ == '__main__':
    print(get_available_datasets(return_details=True))
    for X, y in load_dataset():
        print(f'{X.shape}, {y.shape}')
