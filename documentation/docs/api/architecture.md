# Package Structure
The photonai source code is divided in the following folders

* _base_:
  Here reside photonai's core elements such as the Hyperpipe, the pipeline, the pipeline element and all other photonai pipeline specialities.
* _helper_: not much to say here
* _modelwrapper_:
  All algorithms shipped with PHOTONAI and wrappers for accessing non-scikit-learn conform algorithms are stored here. 
* _optimization_:
  Everything around Hyperparameter Optimization.
* _photonlogger_:
  Special logging logic to make everything as informative and pretty as possible. Also to avoid naming conflicts with
  loggers from other packages.
* _processing_ :  Here reside all classes that do the actual computing
    


