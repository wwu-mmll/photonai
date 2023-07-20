from run_elements import BreastCancerRunner, DiabetesRunner
from config_selectors import DefaultConfigSelector
from collect_results import ResultCollector


list_of_dataset_runners = {'breast_cancer': BreastCancerRunner,
                           'diabetes': DiabetesRunner}

current_config_selector = DefaultConfigSelector
config_selector_name = 'default'

for name, runner_type in list_of_dataset_runners.items():
    # todo: add multiprocessing!
    project_folder = './tmp/' + config_selector_name
    runner = runner_type(name=name,
                         project_folder=project_folder,
                         best_config_selector=DefaultConfigSelector())
    # runner.run_analysis()


collector = ResultCollector(project_folder)
collector.collect_results()
