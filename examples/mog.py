import pandas as pd
import numpy as np
from photonai import PipelineElement, ClassificationPipe, ClassifierSwitch, FloatRange, PermutationTest
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


class MOGAnalysis:

    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.root_path = self.raw_data_path.parent
        self.cleaned_data_path = self.raw_data_path.parent.joinpath(self.raw_data_path.name[:-4] + "_cleaned.csv")
        self.data = None
        self.random_state = 2892
        self.perm_id = 'a26c54a9-28b5-4fb4-8ac8-5de77deb16d2'

    @staticmethod
    def convert_to_float(string_value: str):
        try:
            value = float(string_value)
        except ValueError:
            value = np.nan
        return value

    def clean_data(self):
        # Load data
        df = pd.read_csv(self.raw_data_path)
        df['target'] = df['target'].replace(2, 1)
        df = df[df['target'] != 3]
        df = df.applymap(lambda x: x.replace('> ', '') if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.replace('< ', '') if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.replace('-', "") if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.replace('NaN', "") if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.replace('empty', "") if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.replace(' ', "") if isinstance(x, str) else x)
        df['Titer (Serum)'] = df['Titer (Serum)'].replace('negative', 10)
        df['Titer (Serum)'] = df['Titer (Serum)'].str.split(':').str[1]
        df = df.applymap(lambda x: self.convert_to_float(x))
        df = df.astype(float)
        # too few samples in this column:
        df.drop(columns=["GFAP pg/ml"], inplace=True)
        df.to_csv(self.cleaned_data_path, index=False)

        self.data = df

    def setup_features(self):

        if self.data is None:
            self.data = pd.read_csv(self.cleaned_data_path)

        # clean NaNs in target
        self.data = self.data[~self.data["target"].isna()]
        X = self.data.iloc[:, 1:53].values
        y = self.data.iloc[:, 0].values.astype(int)

        return X, y

    def setup_hyperpipe(self):
        hyperpipe = ClassificationPipe('mog',
                                       project_folder=self.root_path.joinpath('MOG_ML'),
                                       optimizer="random_search",
                                       optimizer_params={'n_configurations': 35},
                                       outer_cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                                                        random_state=self.random_state),
                                       inner_cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                                random_state=self.random_state),
                                       use_test_set=True,
                                       add_default_pipeline_elements=False)

        hyperpipe += PipelineElement('SimpleImputer', missing_values=np.nan)
        hyperpipe += PipelineElement('MinMaxScaler')

        # unfortunately, there are too few samples for SMOTE!
        hyperpipe += PipelineElement('ImbalancedDataTransformer', method_name='RandomUnderSampler')

        estimator_switch = ClassifierSwitch('final_estimator')
        estimator_switch += PipelineElement('LinearDiscriminantAnalysis')
        hyperpipe += estimator_switch
        return hyperpipe

    def ml_analysis(self):
        hyperpipe = self.setup_hyperpipe()
        hyperpipe.fit(*self.setup_features())

        feature_importances = hyperpipe.get_permutation_feature_importances()
        fi_data = np.array(list(feature_importances.values())).transpose()
        fi_df = pd.DataFrame(data=fi_data, index=list(self.data.columns[1:53]),
                             columns=list(feature_importances.keys()))
        # fi_df.sort_values(by=["mean"], inplace=True)
        fi_df.to_csv(self.root_path.joinpath('permutation_feature_importances.csv'))

    def permutation_test(self):
        perm_tester = PermutationTest(self.setup_hyperpipe, n_perms=2, n_processes=1,
                                      random_state=self.random_state, permutation_id=self.perm_id)
        perm_tester.fit(*self.setup_features())

        results = PermutationTest._calculate_results(self.perm_id,
                                                     mongodb_path='mongodb://localhost:27017/photon_results')
        print(results.p_values)


if __name__ == '__main__':
    root_path = Path('/home/m/Projects/paula/')
    file_path = root_path.joinpath('Table_20230906.csv')

    analysis = MOGAnalysis(file_path)
    analysis.clean_data()
    analysis.ml_analysis()
    # analysis.permutation_test()




