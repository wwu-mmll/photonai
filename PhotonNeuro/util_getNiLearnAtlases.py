# Atlas from nilearn are functions ('call'), simple nii maps are also possible (type 'map')
ATLAS_DICT = {#'ICBM': ('call', "datasets.fetch_icbm152_2009()"),
              'AAL': ('call', "datasets.fetch_atlas_aal()"),
              'HarvardOxford25_cort_1mm': ('call', "datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')"),
              'HarvardOxford25_sub_1mm': ('call', "datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')"),
              'HarvardOxford50_cort_1mm': ('call', "datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')"),
              'HarvardOxford50_sub_1mm': ('call', "datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')"),
            #  'MSDL': ('call', "datasets.fetch_atlas_msdl()"),
            #  'Smith': ('call', "datasets.fetch_atlas_smith_2009()"),
            #   '' C:\Users\hahnt / nilearn_data\smith_2009\rsn20.nii.gz
            # C:\Users\hahnt / nilearn_data\smith_2009\PNAS_Smith09_rsn10.nii.gz
            # C:\Users\hahnt / nilearn_data\smith_2009\rsn70.nii.gz
            # C:\Users\hahnt / nilearn_data\smith_2009\bm20.nii.gz
            # C:\Users\hahnt / nilearn_data\smith_2009\PNAS_Smith09_bm10.nii.gz
            # C:\Users\hahnt / nilearn_data\smith_2009\bm70.nii.gz
              # 'Allen': ('call', "datasets.fetch_atlas_allen_2011()")
              }

# #################################################################################
# extract relevant info from nilearn atlases
new_dict = dict()
for atlas_name in ATLAS_DICT:
    print(atlas_name)
    from nilearn import datasets

    atlas_in = eval(ATLAS_DICT[atlas_name][1])
    # load atlas image
    if atlas_name == 'Smith':
        for netName in atlas_in:
            if netName == 'description':
                continue
            atlas_file = atlas_in[netName]
            print(atlas_file)
    else:
        atlas_file = atlas_in['maps']

    map = load_img(atlas_file).get_data()  # get actual map data

    # Use given indices if they exist; otherwise use whatever is in the atlas
    if 'indices' in atlas_in:
        indices = [int(i) for i in atlas_in['indices']]  # convert list of string indices to int
    else:
        indices = list(np.unique(map))
    # Use given labels if they exist; otherwise use map indices
    if 'labels' in atlas_in:
        atlas_labels = atlas_in['labels']
    else:
        atlas_labels = list(str(i) for i in indices)

    # build labels.txt
    labels_file = (atlas_file[:-4] + '_labels.txt')
    print(labels_file)
    content = dict(zip(indices, atlas_labels))
    with open(labels_file, 'w') as f:
        [f.write('{0}\t{1}\n'.format(key, value)) for key, value in content.items()]

    new_dict[atlas_name] = [atlas_file, labels_file]
print(new_dict)

#     #################################################################################
