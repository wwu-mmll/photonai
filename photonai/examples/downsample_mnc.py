from photonai.neuro.ImageBasics import ResampleImages, SmoothImages
import numpy as np
import pickle

file = ["/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0001.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0009.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0020.nii"]

t = ResampleImages(voxel_size=[1, 1, 1], nr_of_processes=3)
resampled_images = t.transform(file)

# # pickle.dump(resampled_images, open('resampled_imgs.p', 'wb'))
#
# resampled_images = pickle.load(open('resampled_imgs.p', 'rb'))
t2 = SmoothImages(fwhm=[2, 2, 2], nr_of_processes=3)
print("Now should smooth images")
smoothed_images = t2.transform(resampled_images)
print("Images smoothed")

print(len(smoothed_images))
t3 = ResampleImages(voxel_size=[3, 5, 10], nr_of_processes=5)
res_smoothed_images = t3.transform(smoothed_images)


# downsampled_file = resample_img(file[0], target_affine=np.diag([1, 1, 1]), interpolation='nearest')
# downsampled_file.to_filename('/spm-data/Scratch/spielwiese_ramona/PAC2019/raw_data/N3107/N3107_downsampled.nii')
debug = True


