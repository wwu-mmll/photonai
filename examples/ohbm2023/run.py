import pandas as pd

from run_elements import *
from config_selectors import DefaultConfigSelector, RandomConfigSelector
from collect_results import ResultCollector
from multiprocessing import Process
import os


list_of_config_selectors = {'default': DefaultConfigSelector,
                            'random': RandomConfigSelector}


config_selector_name = 'default'
multiprocessing = False
calculate = False


list_of_dataset_runners = {
                           'abalone': AbaloneRunner,
                           'habermans_survival': HabermansSurvivalRunner,
                           'autistic': AutisticRunner,
                           'parkinson': ParkinsonsRunner,
                           'breast_cancer': BreastCancerRunner,
                           'diabetes': DiabetesRunner,
}

procs = []
base_project_folder = './tmp/'
for config_selector_name, current_config_selector in list_of_config_selectors.items():
    for name, runner_type in list_of_dataset_runners.items():

        project_folder = base_project_folder + config_selector_name
        os.makedirs(project_folder, exist_ok=True)
        runner = runner_type(name=name,
                             project_folder=project_folder,
                             best_config_selector=current_config_selector())
        func = runner.run_analysis

        if not multiprocessing and calculate:
            func()
        elif multiprocessing and calculate:
            proc = Process(target=func)
            procs.append(proc)
            proc.start()
        else:
            print(f"Calculate is {calculate}")

    if multiprocessing is True:
        for proc in procs:
            proc.join()

    collector = ResultCollector(project_folder)
    collector.collect_results()


regression_results = None
classification_results = None

for config_selector_name, _ in list_of_config_selectors.items():
    project_folder = base_project_folder + config_selector_name
    classification = pd.read_csv(project_folder + '/classification.csv')
    classification['config_selector_name'] = config_selector_name
    if classification_results is None:
        classification_results = classification
    else:
        classification_results = pd.concat([classification_results, classification], ignore_index=True)
    regression = pd.read_csv(project_folder + '/regression.csv')
    regression['config_selector_name'] = config_selector_name
    if regression_results is None:
        regression_results = regression
    else:
        regression_results = pd.concat([regression_results, regression], ignore_index=True)

regression_results.to_csv(base_project_folder+"/regression.csv")
classification_results.to_csv(base_project_folder+"/classification.csv")
