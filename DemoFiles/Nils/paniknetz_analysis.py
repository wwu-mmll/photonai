#-----------------------------------------------------------------------
# # Treatment Response Prediction of CBT for Panic Disorder
# using SNP whole genome data
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# Load data
#-----------------------------------------------------------------------

import numpy as np
import pandas as pd

# Load first batch (1. Förderperiode des BMBF). Data is a rectangular
# matrix with subjects times variants. Variants are coded as the
# frequency of alternate alleles. Created with PSEQ (project: paniknetz_b1.pseq).
data = pd.read_table('/media/HDD1/paniknetz/panic_data/panik_batch1_v_matrix.txt')
data = data.transpose()
data = data.drop(['VAR','REF','ALT'])

# Read pheno data from excel file (same folder as other data and pseq project).
pheno = pd.read_excel('/media/HDD1/paniknetz/panic_data/Phänodaten.xlsx')

# Get only most important variables and just one treatment response
# outcome variable (resp_HAMA_pa)
# 1 = Responder im total HAMA-Score bei Post, mindestens 50% Verringerung
#  seit Eingangsuntersuchung, 0 = kein Responder
# Group defines experimental group: 1 begleitet, 2 unbegleitet, 3 warteliste

pheno = pheno[['FID','sex','age','group','resp_HAMA_pa']]

# ...drop all nans
pheno = pheno.dropna()

# loop through all subjects within the "pheno" dataframe, search for
# the ID in "data" and save the SNPs, the group and the outcome variable...
mat = []
targets = []
group = []
for i in range(pheno.shape[0]):
    try:
        mat.append(data.loc[pheno.iloc[i,0]])
        targets.append(pheno.iloc[i,4])
        group.append(pheno.iloc[i,3])
    except:
        pass

# ...stack and make numpy array
X = np.stack(mat)
targets = np.asarray(targets)
group = np.asarray(group)
del data
del mat

#-----------------------------------------------------------------------
# Run PHOTON
#-----------------------------------------------------------------------

from sklearn.model_selection import KFold
from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(np.reshape(targets,(targets.shape[0],1)))
y = enc.transform(np.reshape(targets,(targets.shape[0],1))).toarray()

# create cross-validation object first
outer_cv = KFold(n_splits=3, shuffle=True, random_state=14)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=15)

# create a hyperPipe
pipe = Hyperpipe('panik', outer_cv= outer_cv, inner_cv=inner_cv,
                 best_config_metric='accuracy',optimizer='random_grid_search',
                 metrics=['accuracy'],verbose=2)

# SVMs (linear and rbf)
svc_estimator = PipelineElement.create('svc', {}, kernel='linear')
dnn_estimator = PipelineElement.create('KerasDNNClassifier',
                                       {'hidden_layer_sizes':[[10, 20]],
                                        'nb_epoch':[100], 'target_dimension':[2]})

# and then user either SVC or DNN
est = PipelineSwitch('switch', [svc_estimator, dnn_estimator])
pipe.add(est)
pipe.fit(X[group==1], y[group==1])