
<h1>Reasons to Use PHOTONAI</h1>
<ul class="uk-list">
<li>It allows researchers to build, optimize and evaluate machine learning pipelines with 
    few lines of code</li>
<li>It automates the training, optimization and test workflow according to user-defined parameters. </li>
<li>It offers convenient access to established machine learning toolboxes such as sklearn</li>
<li>It is especially well-suited for researchers with little programming experience. </li>
<li>Users can select and change hyperparameter optimization strategies by keywords</li>
<li> It is built on clean interfaces and therefore fully customizable.</li>
<li>It is easily extendable with custom algorithms, e.g. for handling biomedical data modalities.</li>
<li>It acts as a unifying framework to help researchers share and reuse code across projects. </li>
<li>It offers both simple and parallel pipeline streams for comparing algorithms, combining features and building ensembles</li>
<li>It extends existing pipeline implementations to enable the developer, e.g. to change the dataset (data augmentation) 
    within the training and testing cross validation splits at runtime.</li>
<li>It enables rapid prototyping in contexts which require iterative evaluation of novel machine learning models.</li>
<li>and many others...</li>
</ul>

![Basic PHOTONAI class hierachy](https://www.photon-ai.com/static/img/architecture.jpg "PHOTONAI class diagram")

## Class structure 
The PHOTONAI framework is built to accelerate and simplify the design of machine learning pipelines and automatize the training, testing and hyperparameter optimization process. The most important class is the _Hyperpipe_, as it is used to parametrize and control both the pipeline and the training and testing workflow. The _Pipeline_ streams data through a sequence of _PipelineElements_, the latter of which represent either established or custom algorithm implementations (_BaseElement_). _PipelineElements_ can share a position within the data stream via an And-Operation (_Stack_), an Or-Operation (_Switch_) or represent a parallel sub-pipeline (_Branch_)