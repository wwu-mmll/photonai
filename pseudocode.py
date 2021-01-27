pipeline = initalize_pipeline()
data = sanity_check_input_data()

for outer_fold in outer_folds:
    outer_fold_data = outer_cv_strategy.get(outer_fold, data)

    # apply most trivial prediction strategy to estimate baseline performance
    run_dummy_estimator(outer_fold_data)

    # initialize hyperparameter optimization space
    hyperparameter_optimizer.prepare(pipeline)

    # ask hyperparameter optimization strategy
    # for next hyperparameter configuration
    for hp_config in hyperparameter_optimizer.ask():
        for inner_fold in inner_folds:
            inner_fold_data = inner_cv_strategy.get(inner_fold,
                                                    outer_fold_data)
            # train and evaluate on validation set
            current_performance = train_and_test_pipeline(hp_config,
                                                          pipeline,
                                                          inner_fold_data)

            # inform hyperparameter optimization strategy
            hyperparameter_optimizer.tell(current_performance)

            if performance_constraints:
                # check if hp_config shall be further evaluated
                # or is dismissed due to bad performance
                if not current_performance > performance_constraints:
                    break

        # log best hyperparameter configuration so far
        if current_performance > best_performance:
            best_performance = current_performance
            best_config = hp_config

    best_configs_outer_folds.append(best_config)

    # evaluate performance of best configuration on test set
    train_and_test_pipeline(best_config,
                            pipeline,
                            outer_fold_data)

# select overall best config across best configs of outer folds
overall_best_config = max_performance(best_configs_outer_folds)
# setup pipeline with best config
pipeline.set_params(overall_best_config)
# train with all data
pipeline.metrics_train(data)
# save the final model in a standardized format
pipeline.save()
