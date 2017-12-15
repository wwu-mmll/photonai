from sklearn.datasets import load_boston
import numpy as np
from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch
from sklearn.model_selection import ShuffleSplit

boston = load_boston()
y1 = boston.target
y2 = np.random.randn(y1.shape[0])  #add other targets
y3 = 3*np.random.randn(y1.shape[0])

multi_y = [y1,y2,y3]
X = boston.data

# create list of dictionaries that define outputs
outputs = []
names = ['targets1', 'targets2', 'targets3']
for i in range(len(multi_y)):
    output_node_dict = {'name':names[i], 'target_dimension':1, 'activation':'linear',
                        'loss':'mse', 'loss_weight':1}
    outputs.append(output_node_dict)


# Define hyperpipe
cv = ShuffleSplit(n_splits=1,test_size=0.2)

pipe = Hyperpipe('pipe', optimizer='grid_search',
                     metrics=['score'],
                     inner_cv=cv,
                     eval_final_performance=True, verbose=2)



pipe += PipelineElement.create('KerasDNNMultiOutput',
                               {'hidden_layer_sizes':[[50,20]],
                                'dropout_rate':[0, .5, .9],
                                'nb_epoch':[200],
                                'act_func':['relu', 'sigmoid'],
                                'learning_rate': [.1 .01 .001],
                                'batch_normalization':[True],
                                'early_stopping_flag':[True],
                                'eaSt_patience':[20],
                                'reLe_factor':[0.4],
                                'reLe_patience':[5]},
                               scoring_method='mean_absolute_error',
                               list_of_outputs=outputs)

y_array = np.transpose(np.asarray(multi_y))
pipe.fit(X,y_array)
results_tree = pipe.result_tree
print(results_tree.get_best_config_for(outer_cv_fold=0))
print('')




