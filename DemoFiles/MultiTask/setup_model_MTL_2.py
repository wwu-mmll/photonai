# setup photon HP for Multi Task Learning
def setup_model_MTL_2(target_info):
    # import PHOTON Core
    from Framework.PhotonBase import PipelineElement, PipelineSwitch, Hyperpipe, ShuffleSplit
    from sklearn.model_selection import KFold

    metrics = ['variance_explained']
    metrics = ['mean_squared_error']
    #cv = KFold(n_splits=20, shuffle=True, random_state=3)
    #cv = ShuffleSplit(n_splits=1, test_size=0.2)
    cv = KFold(n_splits=3, shuffle=True, random_state=14)
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
    my_pipe += PipelineElement.create('interaction_terms', {'degree': [2]},  interaction_only=True, include_bias=False, test_disabled=True)

    # define Multi-Task-Learning Model
    my_pipe += PipelineElement.create('KerasDNNMultiOutput',
                                      {#'hidden_layer_sizes': [[2048, 512, 256]],
                                       #'dropout_rate': [.5],
                                       #'hidden_layer_sizes': [[256, 16, 16, 16, 16, 8, 8, 8], [256, 16, 16, 16, 16], [256, 16, 8], [256, 16], [32, 16], [32]],
                                       'hidden_layer_sizes': [[32, 4]],
                                       'dropout_rate': [0],
                                       'nb_epoch': [2000],
                                       'act_func': ['sigmoid'],
                                       'learning_rate': [.001],
                                       'batch_normalization': [True],
                                       'early_stopping_flag': [True],
                                       'eaSt_patience': [100],
                                       'reLe_factor': [0.4],
                                       'reLe_patience': [20],
                                       'use_spacecraft_loss': [False]},
                                      scoring_method=metrics[0],
                                      list_of_outputs=target_info,
                                      batch_size=16)
    # my_pipe += PipelineElement.create('KerasDNNMultiOutput',
    #                                {'hidden_layer_sizes': [[5], [10, 5]],
    #                                 'dropout_rate': [0, .2, .5, .8],
    #                                 'nb_epoch': [1000],
    #                                 'act_func': ['relu', 'sigmoid'],
    #                                 'learning_rate': [.01, .1],
    #                                 'batch_normalization': [True],
    #                                 'early_stopping_flag': [True],
    #                                 'eaSt_patience': [50],
    #                                 'reLe_factor': [0.4],
    #                                 'reLe_patience': [5]},
    #                            scoring_method=metrics[0],
    #                            list_of_outputs=target_info, use_spacecraft_loss=True)
    # my_pipe += PipelineElement.create('KerasDNNMultiOutput',
    #                                {'hidden_layer_sizes': [[5], [10], [100], [100, 10], [50, 20], [50, 20, 5]],
    #                                 'dropout_rate': [0, .2, .7],
    #                                 'nb_epoch': [1000],
    #                                 'act_func': ['relu', 'sigmoid'],
    #                                 'learning_rate': [0.01],
    #                                 'batch_normalization': [True],
    #                                 'early_stopping_flag': [True],
    #                                 'eaSt_patience': [20],
    #                                 'reLe_factor': [0.4],
    #                                 'reLe_patience': [5]},
    #                            scoring_method=metrics[0],
    #                            list_of_outputs=target_info, use_spacecraft_loss=False)

    return my_pipe, metrics
