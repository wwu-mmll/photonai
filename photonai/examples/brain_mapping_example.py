from photonai.neuro.AtlasStacker import AtlasInfo
from photonai.neuro.AtlasMapping import AtlasMapping

def hyperpipe_constructor():
    # hyperpipe construtor
    from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
    from sklearn.model_selection import KFold

    pers_opts = PersistOptions(local_file='dummy_file',
                               save_predictions='best',
                               save_feature_importances='None')

    my_pipe = Hyperpipe(name='dummy_pipe',  # the name of your pipeline
                        optimizer='grid_search',  # which optimizer PHOTON shall use
                        metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],
                        best_config_metric='mean_absolute_error',
                        outer_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                        inner_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                        persist_options=pers_opts,
                        verbosity=0)

    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('SVR', {'kernel': ['linear', 'rbf']})

    return my_pipe


if __name__ == '__main__':
    # get oasis gm data and age from nilearn; imgs
    from nilearn import datasets
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=90)
    dataset_files = oasis_dataset.gray_matter_maps
    targets = oasis_dataset.ext_vars['age'].astype(float)  # age

    # which atlases are available?
    # BrainAtlas.whichAtlases()

    # where to write the results
    folder = ''

    # get info for the atlas
    atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Precentral_L', 'Precentral_R', 'Frontal_Sup_R'],
                           extraction_mode='vec')
    #atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='box')


    atlas_mapper = AtlasMapping(atlas_info=atlas_info, hyperpipe_constructor=hyperpipe_constructor,
                                write_to_folder=folder,
                                n_processes=3, write_summary_to_excel=True)

    results = atlas_mapper.fit(dataset_files=dataset_files, targets=targets)
    print(results)
