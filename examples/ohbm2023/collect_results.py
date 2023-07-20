import pandas as pd
import numpy as np
from pathlib import Path
from photonai.processing import ResultsHandler


class ResultCollector:

    def __init__(self, result_folder):
        self.output_path = Path(result_folder)

    def collect_results(self):
        for analysis_type in ['classification', 'regression']:
            result_df = pd.DataFrame()
            for analysis_path in self.output_path.joinpath(analysis_type).iterdir():
                if analysis_path.is_dir():
                    result_path = next(analysis_path.iterdir())
                    print(result_path)

                    res_handler = ResultsHandler()
                    res_handler.load_from_file(result_path.joinpath('photonai_results.json'))
                    perf = res_handler.get_performance_outer_folds()
                    perf_dict = {}
                    for metric_name, metric_values in perf.items():
                        perf_dict[metric_name + '_mean'] = [np.mean(metric_values)]
                        perf_dict[metric_name + '_std'] = [np.std(metric_values)]
                    result_df = pd.concat([result_df, pd.DataFrame(perf_dict)], ignore_index=True)
            result_df.to_csv(self.output_path.joinpath(analysis_type + ".csv"), index=False)

