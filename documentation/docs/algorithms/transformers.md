{% set items = { 'Preprocessing': [
                { 'module': 'Binarizer', 'class':'sklearn.preprocessing.Binarizer', 'package': 'scikit-learn'},
                { 'module': 'FeatureEncoder', 'class': 'photonai.modelwrapper.OrdinalEncoder.FeatureEncoder',
                    'package': 'PHOTON'},
                { 'module': 'FunctionTransformer', 'class': 'sklearn.preprocessing.FunctionTransformer',
                    'package': 'scikit-learn'},
                { 'module': 'KernelCenterer', 'class': 'sklearn.preprocessing.KernelCenterer',
                    'package': 'scikit-learn'},
                { 'module': 'LabelEncoder', 'class': 'sklearn.preprocessing.LabelEncoder', 'package': 'scikit-learn',
                    'example': 'classification_basic'},
                { 'module': 'MaxAbsScaler', 'class': 'sklearn.preprocessing.MaxAbsScaler', 'package': 'scikit-learn'},
                { 'module': 'MinMaxScaler', 'class': 'sklearn.preprocessing.MinMaxScaler', 'package': 'scikit-learn',
                    'example': 'subpipelines'},
                { 'module': 'Normalizer', 'class': 'sklearn.preprocessing.Normalizer', 'package': 'scikit-learn'},
                { 'module': 'PolynomialFeatures', 'class': 'sklearn.preprocessing.PolynomialFeatures',
                    'package': 'scikit-learn'},
                { 'module': 'QuantileTransformer', 'class': 'sklearn.preprocessing.QuantileTransformer',
                    'package': 'scikit-learn', 'example': 'subpipelines'},
                { 'module': 'RobustScaler', 'class': 'sklearn.preprocessing.RobustScaler', 'package': 'scikit-learn'},
                { 'module': 'SimpleImputer', 'class': 'sklearn.impute.SimpleImputer', 'package': 'scikit-learn'},
                { 'module': 'StandardScaler', 'class': 'sklearn.preprocessing.StandardScaler',
                    'package': 'scikit-learn', 'example': 'classification_basic'},
                { 'module': 'SourceSplitter', 'class': 'photonai.modelwrapper.source_splitter.SourceSplitter',
                     'package': 'PHOTONAI'}
             ],

                'Decomposition': [
                { 'module': 'CCA', 'class': 'sklearn.cross_decomposition.CCA', 'package': 'scikit-learn'},
                { 'module': 'DictionaryLearning', 'class': 'sklearn.decomposition.DictionaryLearning',
                    'package': 'scikit-learn'},
                { 'module': 'dict_learning', 'class': 'sklearn.decomposition.dict_learning', 'package': 'scikit-learn'},
                { 'module': 'dict_learning_online', 'class': 'sklearn.decomposition.dict_learning_online',
                    'package': 'scikit-learn'},
                { 'module': 'FactorAnalysis', 'class': 'sklearn.decomposition.FactorAnalysis',
                    'package': 'scikit-learn'},
                { 'module': 'FastICA', 'class': 'sklearn.decomposition.FastICA', 'package': 'scikit-learn'},
                { 'module': 'IncrementalPCA', 'class': 'sklearn.decomposition.IncrementalPCA',
                    'package': 'scikit-learn'},
                { 'module': 'KernelPCA', 'class': 'sklearn.decomposition.KernelPCA', 'package': 'scikit-learn'},
                { 'module': 'LatentDirichletAllocation', 'class': 'sklearn.decomposition.LatentDirichletAllocation',
                    'package': 'scikit-learn'},
                { 'module': 'MiniBatchDictionaryLearning', 'class': 'sklearn.decomposition.MiniBatchDictionaryLearning',
                    'package': 'scikit-learn'},
                { 'module': 'MiniBatchSparsePCA', 'class': 'sklearn.decomposition.MiniBatchSparsePCA',
                    'package': 'scikit-learn'},
                { 'module': 'NMF', 'class': 'sklearn.decomposition.NMF', 'package': 'scikit-learn'},
                { 'module': 'PCA', 'class': 'sklearn.decomposition.PCA', 'package': 'scikit-learn',
                    'example': 'classification_basic'},
                { 'module': 'PLSCanonical', 'class': 'sklearn.cross_decomposition.PLSCanonical', 'package': 'scikit-learn'},
                { 'module': 'PLSRegression', 'class': 'sklearn.cross_decomposition.PLSRegression', 'package': 'scikit-learn'},
                { 'module': 'PLSSVD', 'class': 'sklearn.cross_decomposition.PLSSVD', 'package': 'scikit-learn'},
                { 'module': 'SparsePCA', 'class': 'sklearn.decomposition.SparsePCA', 'package': 'scikit-learn'},
                { 'module': 'SparseCoder', 'class': 'sklearn.decomposition.SparseCoder', 'package': 'scikit-learn'},
                { 'module': 'TruncatedSVD', 'class': 'sklearn.decomposition.TruncatedSVD', 'package': 'scikit-learn'},
                { 'module': 'sparse_encode', 'class': 'sklearn.decomposition.sparse_encode', 'package': 'scikit-learn'},
                ],

                'Feature Selection':[
                { 'module': 'FClassifSelectPercentile', 'class': 'photonai.modelwrapper.FeatureSelection.FClassifSelectPercentile', 'package': 'PHOTONAI'},
                { 'module': 'FRegressionFilterPValue', 'class': 'photonai.modelwrapper.FeatureSelection.FRegressionFilterPValue', 'package': 'PHOTONAI'},
                { 'module': 'FRegressionSelectPercentile', 'class': 'photonai.modelwrapper.FeatureSelection.FRegressionSelectPercentile', 'package': 'PHOTONAI'},
                { 'module': 'GenericUnivariateSelect', 'class': 'sklearn.feature_selection.GenericUnivariateSelect', 'package': 'scikit-learn'},
                { 'module': 'LassoFeatureSelection',
                    'class': 'photonai.modelwrapper.FeatureSelection.LassoFeatureSelection', 'package': 'PHOTONAI',
                    'example': 'classifier_ensemble'},
                { 'module': 'RFE', 'class': 'sklearn.feature_selection.RFE', 'package': 'scikit-learn'},
                { 'module': 'RFECV', 'class': 'sklearn.feature_selection.RFECV', 'package': 'scikit-learn'},
                { 'module': 'SelectPercentile', 'class': 'sklearn.feature_selection.SelectPercentile', 'package': 'scikit-learn'},
                { 'module': 'SelectKBest', 'class': 'sklearn.feature_selection.SelectKBest', 'package': 'scikit-learn'},
                { 'module': 'SelectFpr', 'class': 'sklearn.feature_selection.SelectFpr', 'package': 'scikit-learn'},
                { 'module': 'SelectFdr', 'class': 'sklearn.feature_selection.SelectFdr', 'package': 'scikit-learn'},
                { 'module': 'SelectFromModel', 'class': 'sklearn.feature_selection.SelectFromModel', 'package': 'scikit-learn'},
                { 'module': 'SelectFwe', 'class': 'sklearn.feature_selection.SelectFwe', 'package': 'scikit-learn'},
                { 'module': 'VarianceThreshold', 'class': 'sklearn.feature_selection.VarianceThreshold', 'package': 'scikit-learn'},
                ],

                'Other':[
                { 'module': 'ConfounderRemoval', 'class': 'photonai.modelwrapper.ConfounderRemoval.ConfounderRemoval',
                    'package': 'PHOTONAI', 'example': 'confounder_removal'},
                { 'module': 'ImbalancedDataTransformer',
                    'class': 'photonai.modelwrapper.imbalanced_data_transformer.ImbalancedDataTransformer',
                    'package': 'imbalanced-learn / PHOTONAI', 'example': 'imbalanced_data'},
                { 'module': 'SamplePairingClassification',
                    'class': 'photonai.modelwrapper.SamplePairing.SamplePairingClassification', 'package': 'PHOTONAI',
                    'example': 'sample_pairing'},
                { 'module': 'SamplePairingRegression', 'class': 'photonai.modelwrapper.SamplePairing.SamplePairingRegression', 'package': 'PHOTONAI'},
                ]
            }
         %}

{% set sklearn_path = "https://scikit-learn.org/stable/modules/generated/" %}
{% set github_path = "https://github.com/wwu-mmll/photonai/blob/master/" %}

<h1>Transformer</h1>
<div class="photon-docu-header">
    {% for key in ['Decomposition', 'Feature Selection', 'Preprocessing', 'Other'] %}
        <div>
            <h2>{{key}}</h2>
            <table class="styled-table">
                <thead>
                    <td>Name</td>
                    <td>Class</td>
                    <td>Package</td>
                </thead>
                <tbody>
                    {% for element in items[key] %}
                        <tr>
                            <td>{{element['module']}}</td>
                            {% if element['package'] == 'scikit-learn'%}
                                <td><a href="{{sklearn_path+element['class']}}">{{element['class']}}</a></td>
                            {% else  %}
                                <td><a href="{{github_path+'/'.join(element['class'].split('.')[:-1])+'.py'}}">{{element['class']}}</a></td>
                            {% endif %}
                            <td>{{element['package']}}</td>
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    {% endfor %}
</div>