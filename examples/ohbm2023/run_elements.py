from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from photonai.base import ClassificationPipe, RegressionPipe
from pathlib import Path
from testsuite import load_dataset


class Runner:

    def __init__(self, name: str,
                project_folder: str = './default',
                best_config_selector: object = None):
        self.name = name
        self.project_folder = Path(project_folder)
        self.best_config_selector = best_config_selector

    def analysis_type(self):
        raise NotImplementedError("Must be overriden by child!")

    def load_data(self):
        raise NotImplementedError("Must be overriden by child!")

    def project_folder_generator(self):
        return self.project_folder.joinpath(self.analysis_type()).joinpath(self.name)

    def define_hyperpipe(self):
        pipe_obj = ClassificationPipe if self.analysis_type() == 'classification' else RegressionPipe
        hyperpipe = pipe_obj(name=self.name,
                             project_folder=self.project_folder_generator(),
                             select_best_config_delegate=self.best_config_selector,
                             imputation=True)
        return hyperpipe

    def run_analysis(self):
        X, y = self.load_data()
        pipe = self.define_hyperpipe()
        pipe.fit(X, y)


class BreastCancerRunner(Runner):

    def load_data(self):
        return load_breast_cancer(return_X_y=True)

    def analysis_type(self):
        return 'classification'


class DiabetesRunner(Runner):

    def load_data(self):
        return load_diabetes(return_X_y=True)

    def analysis_type(self):
        return 'regression'


class AbaloneRunner(Runner):
    def load_data(self):
        # X, y = load_dataset('Abalone')
        return load_dataset('Abalone')
    
    def analysis_type(self):
        return 'classification'


class HabermansSurvivalRunner(Runner):
    def load_data(self):
        return load_dataset("Haberman's Survival")
    
    def analysis_type(self):
        return 'classification'


class AutisticRunner(Runner):
    def load_data(self):
        X, y = load_dataset('Autistic Spectrum Disorder Screening Data for Children')
        return X, y

    def analysis_type(self):
        return 'classification'


class ParkinsonsRunner(Runner):
    def load_data(self):
        X, y = load_dataset("Parkinsons Telemonitoring Data Set")
        # the first row is the header?
        X = X[1::]
        y = y[1::]
        return X, y
    
    def analysis_type(self):
        return 'regression'


