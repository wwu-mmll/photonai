from run_elements import BreastCancerRunner, DiabetesRunner
from config_selectors import DefaultConfigSelector


list_of_dataset_runners = {'breast_cancer': BreastCancerRunner,
                           'diabetes': DiabetesRunner}

current_config_selector = DefaultConfigSelector

for name, runner_type in list_of_dataset_runners.items():
    # todo: add multiprocessing!
    runner = runner_type(name=name,
                         project_folder='./tmp/default',
                         best_config_selector=DefaultConfigSelector())
    runner.run_analysis()

