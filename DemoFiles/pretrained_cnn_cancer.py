import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe
from sklearn.model_selection import ShuffleSplit
import numpy as np
from PIL import Image
import glob, os

# X_train = np.random.rand(100,224,224,3)
# y_train = np.random.randint(0,10,100)
# y_train = tfu.oneHot(y_train)
cv = ShuffleSplit(n_splits=1,test_size=0.2, random_state=23)

# # Resize images and save them
# cnt = 0
# for infile in glob.glob("/spm-data/Scratch/spielwiese_claas/ISIC-images/ISIC_MSK-1_1/ISIC_00114*.jpg"):
#     cnt += 1
#     file, ext = os.path.splitext(infile)
#     im = Image.open(infile)
#     img_resized = im.resize((299,299))
#     file = '/spm-data/Scratch/spielwiese_claas/ISIC-images-small/'
#     img_resized.save(file + str(cnt) + "_299_299", "PNG")

# Load resized images and stack them
data = []
for infile in glob.glob("/spm-data/Scratch/spielwiese_claas/ISIC-images-small/*"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
    data.append(im_arr)

X_train = np.asarray(data)
y_train = np.random.randint(0,2,X_train.shape[0])
y_train = tfu.oneHot(y_train)

#cv = KFold(n_splits=5, random_state=23)
my_pipe = Hyperpipe('mnist_siamese_net', optimizer='grid_search',
                            metrics=['categorical_accuracy'], best_config_metric='categorical_accuracy',
                            hyperparameter_specific_config_cv_object=cv,
                            hyperparameter_search_cv_object=cv,
                            eval_final_performance=True, verbose=2)
#my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('PretrainedCNNClassifier', {'input_shape': [(299,299,3)],'target_dimension': [2],  'nb_epoch':[100], 'size_additional_layer':[100], 'freezing_point':[0]})
my_pipe.fit(X_train,y_train)
