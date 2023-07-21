from run_elements import *
from config_selectors import DefaultConfigSelector, RandomConfigSelector, RankingConfigSelector, WeightedRankingConfigSelector
from collect_results import ResultCollector
from multiprocessing import Process
import os


list_of_config_selectors = {'default': DefaultConfigSelector,
                            'random': RandomConfigSelector,
                            'rank': RankingConfigSelector, 
                            'weighted_rank': WeightedRankingConfigSelector}

config_selector_name = 'rank'
multiprocessing = False

list_of_dataset_runners = {
                           # 'abalone': AbaloneRunner,
                           # 'habermans_survival': HabermansSurvivalRunner,
                           # 'autistic': AutisticRunner,
                           # 'parkinson': ParkinsonsRunner,
                           #'breast_cancer': BreastCancerRunner,
                           'diabetes': DiabetesRunner,
}

current_config_selector = list_of_config_selectors[config_selector_name]

procs = []

for name, runner_type in list_of_dataset_runners.items():

    project_folder = './tmp/' + config_selector_name
    os.makedirs(project_folder, exist_ok=True)
    runner = runner_type(name=name,
                         project_folder=project_folder,
                         best_config_selector=current_config_selector())
    func = runner.run_analysis

    if multiprocessing is False:
        func()
    else:
        proc = Process(target=func)
        procs.append(proc)
        proc.start()

if multiprocessing is True:
    for proc in procs:
        proc.join()

collector = ResultCollector(project_folder)
collector.collect_results()
