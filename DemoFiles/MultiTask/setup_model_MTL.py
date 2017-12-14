# setup photon HP for Multi Task Learning
def setup_model_MTL(target_info):
    # import PHOTON Core
    from Framework.PhotonBase import PipelineElement, PipelineSwitch, Hyperpipe, ShuffleSplit
    from sklearn.model_selection import KFold

    metrics = ['variance_explained', 'pearson_correlation', 'mean_absolute_error']
    #cv = KFold(n_splits=20, shuffle=True, random_state=3)
    cv = ShuffleSplit(n_splits=1, test_size=0.2)

    my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                        optimizer_params={},
                        metrics=['score'],
                        inner_cv=cv,
                        eval_final_performance=False,
                        verbose=2)

    # get interaction terms
    # # register elements
    # from Framework.Register import RegisterPipelineElement
    # photon_package = 'PhotonCore'  # where to add the element
    # photon_name = 'interaction_terms'  # element name
    # class_str = 'sklearn.preprocessing.PolynomialFeatures'  # element info
    # element_type = 'Transformer'  # element type
    # RegisterPipelineElement(photon_name=photon_name,
    #                         photon_package=photon_package,
    #                         class_str=class_str,
    #                         element_type=element_type).add()
    # add the elements
    my_pipe += PipelineElement.create('interaction_terms', {'degree': [2, 3]},  interaction_only=True, include_bias=False, test_disabled=True)

    # define Multi-Task-Learning Model
    my_pipe += PipelineElement.create('KerasDNNMultiOutput',
                                   {'hidden_layer_sizes': [[50, 20]],
                                    'dropout_rate': [0.5],
                                    'nb_epoch': [10],
                                    'act_func': ['relu'],
                                    'learning_rate': [0.01],
                                    'batch_normalization': [True],
                                    'early_stopping_flag': [True],
                                    'eaSt_patience': [20],
                                    'reLe_factor': [0.4],
                                    'reLe_patience': [5]},
                               scoring_method='variance_explained',
                               list_of_outputs=target_info)

    return my_pipe, metrics
