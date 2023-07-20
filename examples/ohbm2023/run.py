from run_elements import *
from config_selectors import DefaultConfigSelector, RandomConfigSelector
from collect_results import ResultCollector
from multiprocessing import Process
import os


list_of_dataset_runners = {'breast_cancer': BreastCancerRunner,}
                           # 'diabetes': DiabetesRunner,
                           # 'abalone': AbaloneRunner,
                           # 'habermans_survival': HabermansSurvivalRunner,
                           # 'autistic': AutisticRunner,
                           # 'parkinson': ParkinsonsRunner}


current_config_selector = RandomConfigSelector
config_selector_name = 'random'
procs = []

for name, runner_type in list_of_dataset_runners.items():
    # todo: add multiprocessing!
    project_folder = './tmp/' + config_selector_name
    os.makedirs(project_folder, exist_ok=True)
    runner = runner_type(name=name,
                         project_folder=project_folder,
                         best_config_selector=current_config_selector())

    func = runner.run_analysis
    func()
    #proc = Process(target=func)
    #procs.append(proc)
    #proc.start()


# complete the processes
#for proc in procs:
#    proc.join()


collector = ResultCollector(project_folder)
collector.collect_results()
