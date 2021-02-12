<h1>How to start</h1>

<h2>Installation</h2>
<p class="small-p">You only need two things: Python 3 and your favourite Python IDE to get started. Then simply install via pip.</p>

```python
# pip install photonai
```

<h2>Build your pipeline</h2>
<div class="tutorial-step">
    <p><span class="bold-paragraph-index">1. Setup New Analysis</span>
        Start by creating a new Hyperpipe instance, naming the analysis and specifying where to save all outputs. 

```python
pipe = Hyperpipe('basic_pipe', project_folder='path/to/project')
```

</div>
<div class="tutorial-step">
    <p><span class="bold-paragraph-index">2. Customize the training and testing process</span>
        Now you can select parameters to customize the training, hyperparameter optimization and testing procedure.
        Particularly, you can choose the hyperparameter optimization strategy, set parameters, choose performance metrics
        and choose the performance metric to minimize or maximize, respectively.
    </p>
</div>
```python 
pipe = Hyperpipe('basic_pipe',
                  output_settings=OutputSettings(project_folder='path/to/project'))

                  # choose hyperparameter optimization strategy
                  optimizer='random_grid_search',

                  # PHOTONAI automatically calculates your preferred metrics
                  metrics=['mean_squared_error', 'pearson_correlation',
                           'mean_absolute_error', 'explained_variance'],

                  # this metrics selects the best hyperparameter configuration
                  # in this case mean squared error is minimized
                  best_config_metric='mean_squared_error',

                  # select cross validation strategies
                  outer_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                  inner_cv=KFold(n_splits=10),
``` 

<div class="tutorial-step">
    <p><span class="bold-paragraph-index">3. Build custom pipeline</span>
        Select and arrange normalization, dimensionality reduction, feature selection, data augmentation,
        over- or undersampling algorithms in simple or parallel data streams. You can integrate
        custom algorithms or choose from our wide range of pre-registered algorithms from established toolboxes.
    </p>
</div>
```python
# access the scikit-learn implementations via keywords
# at the same time define the hyperparameters to optimize for each element
pipe += PipelineElement('StandardScaler')

pipe += PipelineElement('PCA',
                        hyperparameters={'n_components': FloatRange(0.5, 0.8, step=0.1)})

pipe += PipelineElement('RandomForestRegressor',
                        hyperparameters={'n_samples_split': IntegerRange(2, 10)})
```

<div class="tutorial-step">
    <p><span class="bold-paragraph-index">4. Feed Data and Train</span>
        Load your data and start the (nested-) cross-validated hyperparameter optimization, training and evaluation procedure.
    You will see an extensive output to monitor the hyperparameter optimization progress, see the results and track the
    best performances so far.
    </p>
</div>

```python
X, y = load_boston(return_X_y=True)
pipe.fit(X, y)
```

<div class="tutorial-step">
    <p><span class="bold-paragraph-index">5. Visualize Results</span>
        Finally, you find all results conveniently stored in the output folder. You have a final model which is trained
    with the best hyperparameter configuration found to share and apply to new data. In addition, you find a .json
    file where all results from the training, optimization and testing procedure are stored. Take that file and
        drag it into the <a href="https://explorer.photon-ai.com/" target="_blank">Explorer</a>. You will see
    all available information conveniently visualized.
</div>





