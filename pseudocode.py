pipeline = initalize_pipeline()
preprocessed_data = preprocess_data()

for outer_fold in outer_folds:
    outer_fold_data = hyperpipe.outer_cv_strategy.get(outer_fold,
                                                      preprocessed_data)

    # apply most trivial prediction strategy to estimate baseline performance
    run_dummy_estimator(outer_fold_data)

    # initialize hyperparameter optimization strategy
    hyperparameter_optimizer.prepare(pipeline)

    # ask hyperparameter optimization strategy
    # for next hyperparameter configuration
    for hyperparameter_config in hyperparameter_optimizer.ask():
        for inner_fold in inner_folds:
            inner_fold_data = hyperpipe.inner_cv_strategy.get(inner_fold,
                                                              outer_fold_data)
            # train and evaluate on validation set
            current_performance = train_and_test_pipeline(hyperparameter_config,
                                                          pipeline,
                                                          inner_fold_data)
            # log best hyperparameter configuration so far
            if current_performance > best_performance:
                best_performance = current_performance
                best_config = hyperparameter_config

            # inform hyperparameter optimization strategy
            hyperparameter_optimizer.tell(current_performance)
            if not current_performance > hyperpipe.performance_constraints():
                break

    # evaluate performance of best configuration on test set
    train_and_test_pipeline(best_config,
                            pipeline,
                            outer_fold_data)

# train and persist best model
pipeline.set_params(overall_best_config)
pipeline.train(preprocessed_data)
pipeline.save()



