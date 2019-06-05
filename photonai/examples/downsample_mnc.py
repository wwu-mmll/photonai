from photonai.neuro.ImageBasics import ResampleImages, SmoothImages, PatchImages
import numpy as np
import pickle
import time

file = ["/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0001.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0009.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0020.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0070.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0073.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0071.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0072.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0074.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0075.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0076.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0077.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0078.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0080.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0081.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0082.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0083.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0084.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0085.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0086.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0087.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0001.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0009.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0020.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0070.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0073.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0071.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0072.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0074.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0075.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0076.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0077.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0078.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0080.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0081.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0082.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0083.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0084.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0085.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0086.nii",
        "/spm-data/Scratch/spielwiese_ramona/PAC2018/data/PAC2018_0087.nii"
        ]

start_time = time.time()
# t = ResampleImages(voxel_size=[1, 1, 1], nr_of_processes=10)
t = PatchImages(nr_of_processes=10)
resampled_images = t.transform(file)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


start_time = time.time()
# t = ResampleImages(voxel_size=[1, 1, 1], nr_of_processes=1)
t = PatchImages(nr_of_processes=1)
resampled_images = t.transform(file)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# patched_image = PatchImages.draw_patches(resampled_images[0].dataobj, 25)

# patcher = PatchImages()
# patched_images = patcher.transform(resampled_images, nr_of_processes = 10)

debug = True
#
# # # pickle.dump(resampled_images, open('resampled_imgs.p', 'wb'))
# #
# # resampled_images = pickle.load(open('resampled_imgs.p', 'rb'))
# t2 = SmoothImages(fwhm=[2, 2, 2], nr_of_processes=3)
# print("Now should smooth images")
# smoothed_images = t2.transform(resampled_images)
# print("Images smoothed")
#
# print(len(smoothed_images))
# t3 = ResampleImages(voxel_size=[3, 5, 10], nr_of_processes=5)
# res_smoothed_images = t3.transform(smoothed_images)


# downsampled_file = resample_img(file[0], target_affine=np.diag([1, 1, 1]), interpolation='nearest')
# downsampled_file.to_filename('/spm-data/Scratch/spielwiese_ramona/PAC2019/raw_data/N3107/N3107_downsampled.nii')
debug = True


