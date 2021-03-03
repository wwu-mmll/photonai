{% set items = { 'Linear Estimators': [
                { 'module': 'ARDRegression', 'class':'sklearn.linear_model.ARDRegression', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'BayesianRidge', 'class': 'sklearn.linear_model.BayesianRidge', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'ElasticNet', 'class': 'sklearn.linear_model.ElasticNet', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'HuberRegressor', 'class': 'sklearn.linear_model.HuberRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'Lars', 'class': 'sklearn.linear_model.Lars', 'ml_type':'reg', 'package': 'scikit-learn'},
                { 'module': 'Lasso', 'class': 'sklearn.linear_model.Lasso', 'ml_type':'reg', 'package': 'scikit-learn',
                    'example': 'classifier_ensemble'},
                { 'module': 'LassoLars', 'class': 'sklearn.linear_model.LassoLars', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'LinearRegression', 'class': 'sklearn.linear_model.LinearRegression', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'LogisticRegression', 'class': 'sklearn.linear_model.LogisticRegression', 'ml_type':'reg',
                    'package': 'scikit-learn', 'example': 'subpipelines'},
                { 'module': 'PassiveAggressiveClassifier', 'class': 'sklearn.linear_model.PassiveAggressiveClassifier',
                    'ml_type':'class', 'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                { 'module': 'Perceptron', 'class': 'sklearn.linear_model.Perceptron', 'ml_type':'class',
                    'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                { 'module': 'RANSACRegressor', 'class': 'sklearn.linear_model.RANSACRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'Ridge', 'class': 'sklearn.linear_model.Ridge', 'ml_type':'reg', 'package': 'scikit-learn'},
                { 'module': 'RidgeClassifier', 'class': 'sklearn.linear_model.RidgeClassifier', 'ml_type':'class',
                    'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                { 'module': 'SGDClassifier', 'class': 'sklearn.linear_model.SGDClassifier', 'package': 'scikit-learn',
                    'ml_type':'class', 'example': 'classifier_ensemble'},
                { 'module': 'SGDRegressor', 'class': 'sklearn.linear_model.SGDRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'TheilSenRegressor', 'class': 'sklearn.linear_model.TheilSenRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'}
            ],
                'Tree-based': [
                { 'module': 'ExtraTreesClassifier', 'class':'sklearn.ensemble.ExtraTreesClassifier', 'ml_type':'class',
                    'package': 'scikit-learn'},
                { 'module': 'ExtraTreesRegressor', 'class': 'sklearn.ensemble.ExtraTreesRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'DecisionTreeClassifier', 'class': 'sklearn.tree.DecisionTreeClassifier', 'ml_type':'class',
                    'package': 'scikit-learn', 'example': 'regression_basic'},
                { 'module': 'DecisionTreeRegressor', 'class': 'sklearn.tree.DecisionTreeRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn'},
                { 'module': 'RandomForestClassifier', 'class': 'sklearn.ensemble.RandomForestClassifier',
                    'ml_type':'class', 'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                { 'module': 'RandomForestRegressor', 'class': 'sklearn.ensemble.RandomForestRegressor', 'ml_type':'reg',
                    'package': 'scikit-learn', example: 'regression_basic'}
            ],
                'Supported Vector Machines': [
                    { 'module': 'LinearSVC', 'class':'sklearn.svm.LinearSVC', 'ml_type':'class',
                    'package': 'scikit-learn', 'example': 'stack_element'},
                    { 'module': 'LinearSVR', 'class': 'sklearn.svm.LinearSVR', 'ml_type':'reg',
                        'package': 'scikit-learn'},
                    { 'module': 'NuSVC', 'class': 'sklearn.svm.NuSVC', 'ml_type':'class',
                        'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'NuSVR', 'class': 'sklearn.svm.NuSVR', 'ml_type':'reg',
                        'package': 'scikit-learn'},
                    { 'module': 'OneClassSVM', 'class': 'sklearn.svm.OneClassSVM',
                        'ml_type':'class', 'package': 'scikit-learn'},
                    { 'module': 'PhotonOneClassSVM', 'class': 'photonai.modelwrapper.PhotonOneClassSVM.PhotonOneClassSVM',
                        'ml_typy':'reg', 'package': 'scikit-learn / PHOTONAI'},
                    { 'module': 'SVC', 'class': 'sklearn.svm.SVC', 'ml_type':'class',
                        'package': 'scikit-learn', 'example': 'classification_basic'},
                    { 'module': 'SVR', 'class': 'sklearn.svm.SVR', 'ml_type':'reg',
                        'package': 'scikit-learn'},
            ],
                'Neural Networks': [
                    { 'module': 'BernoulliRBM', 'class':'sklearn.neural_network.BernoulliRBM', 'ml_type':'',
                    'package': 'scikit-learn'},
                    { 'module': 'KerasDnnClassifier',
                        'class': 'photonai.modelwrapper.keras_dnn_classifier.KerasDnnClassifier', 'ml_type':'class',
                        'package': 'keras / PHOTONAI', 'example': 'keras_multiclass'},
                    { 'module': 'KerasDnnRegressor',
                        'class': 'photonai.modelwrapper.keras_dnn_regressor.KerasDnnRegressor', 'ml_type':'reg',
                        'package': 'keras / PHOTONAI', 'example': 'keras_basic'},
                    { 'module': 'MLPClassifier', 'class': 'sklearn.neural_network.MLPClassifier', 'ml_type':'class',
                        'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'MLPRegressor', 'class': 'sklearn.neural_network.MLPRegressor',
                        'ml_type':'reg', 'package': 'scikit-learn'},
                    { 'module': 'PhotonMLPClassifier', 'class': 'photonai.modelwrapper.PhotonMLPClassifier.PhotonMLPClassifier',
                        'ml_type':'class', 'package': 'scikit-learn / PHOTONAI', 'example': 'mlp'}
                ],
                'Ensemble': [
                    { 'module': 'AdaBoostClassifier', 'class': 'sklearn.ensemble.AdaBoostClassifier', 'ml_type':'class',
                        'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'AdaBoostRegressor', 'class': 'sklearn.ensemble.AdaBoostRegressor', 'ml_type':'reg',
                        'package': 'scikit-learn'},
                    { 'module': 'BaggingClassifier', 'class': 'sklearn.ensemble.BaggingClassifier',
                        'ml_type':'class', 'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'BaggingRegressor', 'class': 'sklearn.ensemble.BaggingRegressor',
                        'ml_type':'reg', 'package': 'scikit-learn'},
                    { 'module': 'GradientBoostingClassifier', 'class': 'sklearn.ensemble.GradientBoostingClassifier',
                        'ml_type':'class', 'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'GradientBoostingRegressor', 'class': 'sklearn.ensemble.GradientBoostingRegressor',
                        'ml_type':'reg', 'package': 'scikit-learn'},
                ],
                'Neighour-Based': [
                    { 'module': 'KNeighborsClassifier', 'class': 'sklearn.neighbors.KNeighborsClassifier', 'ml_type':'class',
                        'package': 'scikit-learn', 'example': 'subpipelines'},
                    { 'module': 'KNeighborsRegressor', 'class': 'sklearn.neighbors.KNeighborsRegressor', 'ml_type':'reg',
                        'package': 'scikit-learn'},
                    { 'module': 'NearestCentroid', 'class': 'sklearn.neighbors.NearestCentroid',
                        'ml_type':'class', 'package': 'scikit-learn'},
                    { 'module': 'RadiusNeighborsClassifier', 'class': 'sklearn.neighbors.RadiusNeighborsClassifier',
                        'ml_type':'class', 'package': 'scikit-learn'},
                    { 'module': 'RadiusNeighborsRegressor', 'class': 'sklearn.neighbors.RadiusNeighborsRegressor',
                        'ml_type':'reg', 'package': 'scikit-learn'}
                ],
                'Probabilistic': [
                    { 'module': 'BayesianGaussianMixture', 'class': 'sklearn.mixture.BayesianGaussianMixture', 'ml_type':'',
                        'package': 'scikit-learn'},
                    { 'module': 'BernoulliNB', 'class': 'sklearn.naive_bayes.BernoulliNB', 'ml_type':'class',
                        'package': 'scikit-learn'},
                    { 'module': 'GaussianNB', 'class': 'sklearn.naive_bayes.GaussianNB',
                        'ml_type':'class', 'package': 'scikit-learn'},
                    { 'module': 'MultinomialNB', 'class': 'sklearn.naive_bayes.MultinomialNB',
                        'ml_type':'class', 'package': 'scikit-learn'},
                    { 'module': 'GaussianMixture', 'class': 'sklearn.mixture.GaussianMixture',
                        'ml_type':'', 'package': 'scikit-learn'},
                    { 'module': 'GaussianProcessClassifier', 'class': 'sklearn.gaussian_process.GaussianProcessClassifier',
                        'ml_type':'class', 'package': 'scikit-learn', 'example': 'classifier_ensemble'},
                    { 'module': 'GaussianProcessRegressor', 'class': 'sklearn.gaussian_process.GaussianProcessRegressor',
                        'ml_type':'reg', 'package': 'scikit-learn'}
                ],
                'Other': [
                    { 'module': 'DummyClassifier', 'class': 'sklearn.dummy.DummyClassifier', 'ml_type':'class',
                        'package': 'scikit-learn'},
                    { 'module': 'DummyRegressor', 'class': 'sklearn.dummy.DummyRegressor', 'ml_type':'reg',
                        'package': 'scikit-learn'},
                    { 'module': 'KernelRidge', 'class': 'sklearn.kernel_ridge.KernelRidge',
                        'ml_type':'reg', 'package': 'scikit-learn'},
                    { 'module': 'PhotonVotingClassifier', 'class': 'photonai.modelwrapper.Voting.PhotonVotingClassifier',
                        'ml_type':'class', 'package': 'PHOTONAI', 'example': 'classifier_ensemble'},
                    { 'module': 'PhotonVotingRegressor', 'class': 'photonai.modelwrapper.Voting.PhotonVotingRegressor',
                        'ml_type':'reg', 'package': 'PHOTONAI'}
                ],
            }
         %}

{% set sklearn_path = "https://scikit-learn.org/stable/modules/generated/" %}
{% set github_path = "https://github.com/wwu-mmll/photonai/blob/master/" %}
{% set headers = ['Linear Estimators', 'Tree-based', 'Supported Vector Machines',
                    'Neural Networks', 'Ensemble', 'Neighour-Based', 'Probabilistic', 'Other'] %}

<h1>Estimator</h1>

[All](#){: #button_all .md-button--primary .md-button onclick="myFunction('')"}
[Classification](#){: #button_class .md-button onclick="myFunction('class')"}
[Regression](#){: #button_reg .md-button onclick="myFunction('reg')"}

<div class="photon-docu-header">
      {% for i in range(8)%}
        <div>
            {% set key = headers[i] %}
                <h2>{{key}}</h2>
                <div style="transform: scale(1)">
                    <table class="styled-table" data-name=filterTable>
                        <thead>
                            <td>Name</td>
                            <td>Class</td>
                            <td>Package</td>
                        </thead>
                        <tbody>
                            {% for element in items[key] %}
                                <tr ml-type={{element['ml_type']}}>
                                    <td>{{element['module']}}</td>
                                    {% if element['package'] == 'scikit-learn'%}
                                        <td><a href="{{sklearn_path+element['class']}}">{{element['class']}}</a></td>
                                    {% else %}
                                        <td><a href="{{github_path+'/'.join(element['class'].split('.')[:-1])+'.py'}}">{{element['class']}}</a></td>
                                    {% endif %}
                                    <td>{{element['package']}}</td>
                                </tr>
                            {%endfor%}
                        </tbody>
                    </table>
                </div>
        </div>
    {%endfor%}
</div>
    
<script>
function myFunction(filter) {
  var input, table, tr, td, i, txtValue;
  if (filter == 'class') {
    document.getElementById("button_all").classList.remove('md-button--primary');
    document.getElementById("button_class").classList.add('md-button--primary');
    document.getElementById("button_reg").classList.remove('md-button--primary');
  } else if (filter == 'reg') {
    document.getElementById("button_all").classList.remove('md-button--primary');
    document.getElementById("button_class").classList.remove('md-button--primary');
    document.getElementById("button_reg").classList.add('md-button--primary');
  } else {
    document.getElementById("button_all").classList.add('md-button--primary');
    document.getElementById("button_class").classList.remove('md-button--primary');
    document.getElementById("button_reg").classList.remove('md-button--primary');
  }
  alltables = document.querySelectorAll("table[data-name=filterTable]");

  alltables.forEach(function(table){
      tr = table.getElementsByTagName("tr");
      // Loop through all table rows, and hide those who don't match the search query
      for (i = 0; i < tr.length; i++) {
        txtValue = tr[i].getAttribute('ml-type');
        if (txtValue) {
          if (txtValue.indexOf(filter) > -1) {
            tr[i].style.display = "";
          } else {
            tr[i].style.display = "none";
          }
        } 
      }
  });
}
</script>