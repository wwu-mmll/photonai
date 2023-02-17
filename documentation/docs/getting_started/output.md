<h1>PHOTONAI Output</h1>
After executing the script a result folder is created. In there you find six files with different information
about your pipeline and the results.

<h3>photon_summary.txt</h3> 
A text file including a summary of the results.

<h3>best_config_predictions.csv</h3>
This file saves the test set predictions for the best configuration of each outer fold.

<h3>photon_result_file.json</h3>
You can visualize this file with our <a href="https://explorer.photon-ai.com/" target="_blank">Explorer</a>. 

Visualized information:
<ul class="uk-list">
    <li>Best Hyperparameter Configuration</li>
    <li>Performance</li>
    <li>Fold information</li>
    <li>Tested Configuration</li>
    <li>Optimization Progress</li>
</ul>
        
<h3>photon_best_model.photon</h3>
This file stores the best model. You can share or reload it later.

<h3>photon_output.log</h3>
Saves the console output from every fold, including the time, the current testing configurations and the results.

<h3>hyperpipe_config.json</h3>
Here is the initial setup for your analysis, so you can recreate it later.








