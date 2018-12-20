import os
import pandas as pd
import numpy as np
from photonai.base.PhotonBase import Hyperpipe


model_path = '/home/rleenings/Projects/Engima BrainAge/'
# actual_predictions_file = 'best_model_ENIGMA_final_Female.photon'
actual_predictions_file = 'best_model_ENIGMA_final_Male.photon'
csv_file = 'trainControlsMales_raw_agedrop.csv'


# loaded_csv2 = pd.read_csv(os.path.join(model_path, 'trainControlsFemales_raw.csv'))
loaded_csv2 = pd.read_csv(os.path.join(model_path, 'trainControlsMales_raw.csv'))
loaded_csv2 = loaded_csv2.drop(labels=['Age'], axis=1)
loaded_csv2.to_csv(os.path.join(model_path, csv_file), index=False)
#
loaded_csv = pd.read_csv(os.path.join(model_path, csv_file))
my_pipe = Hyperpipe.load_optimum_pipe(os.path.join(model_path, actual_predictions_file))
pipe_predictions = my_pipe.predict(np.squeeze(loaded_csv.values))

debug = True
