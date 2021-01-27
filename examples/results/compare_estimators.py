from photonai.processing import ResultsHandler

res_handler = ResultsHandler()
res_handler.load_from_file("/home/rleenings/tmp/examples/advanced/tmp/heart_failure_no_fu_results_2021-01-22_12-03-03/photon_result_file.json")

output = res_handler.get_best_performances_for_estimator()