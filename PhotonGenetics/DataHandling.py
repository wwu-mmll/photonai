import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp

def convert_to_dose_1(chromo):
    print('Loading chromosome ', chromo)
    inds = pd.read_table(str(path + name_ind_1 + str(chromo) + name_ind_2))
    df = pd.read_csv(str(path + name_chr_1 + str(chromo) + name_chr_2),
             compression='gzip', header=None,  quotechar='"',
            sep='\s+', engine='python')
    x = np.asarray(df.iloc[:,3:].as_matrix().T)
    n = x[0::2,:] + x[1::2,:]*2
    print('Save numpy array...')
    np.save(path + 'chr_' + str(chromo) + '.npy', n)
    del x
    del n
    del df
    return


# path = '/media/HDD1/ConLiGen/ConLiGen/ConLiGen_Imputation/Batch1/HumanOmni1M/'
# name_ind_1 = 'ConLiGen_Batch1_HumanOmni1M_chr'
# name_ind_2 = '_minimac_imputed.ind'
# name_chr_1 = 'ConLiGen_Batch1_HumanOmni1M_chr'
# name_chr_2 = '_minimac_imputed_overlapping.dose.gz'
# p = mp.Pool(3)
# p.map(convert_to_dose_1, range(1,23))


path = '/media/HDD1/ConLiGen/ConLiGen/ConLiGen_Imputation/Batch1/HumanOmni2.5M/'
name_ind_1 = 'ConLiGen_Batch1_HumanOmni2.5M_chr'
name_ind_2 = '_minimac_imputed.ind'
name_chr_1 = 'ConLiGen_Batch1_HumanOmni2.5M_chr'
name_chr_2 = '_minimac_imputed_overlapping.dose.gz'
p = mp.Pool(3)
p.map(convert_to_dose_1, range(1,23))


path = '/media/HDD1/ConLiGen/ConLiGen/ConLiGen_Imputation/Batch1/HumanOmniExpress/'
name_ind_1 = 'ConLiGen_Batch1_HumanOmniExpress_chr'
name_ind_2 = '_minimac_imputed.ind'
name_chr_1 = 'ConLiGen_Batch1_HumanOmniExpress_chr'
name_chr_2 = '_minimac_imputed_overlapping.dose.gz'
p = mp.Pool(3)
p.map(convert_to_dose_1, range(1,23))


