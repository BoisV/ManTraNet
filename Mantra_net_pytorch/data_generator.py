from glob import glob
from operator import mod
from random import random
from PIL import Image
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from scipy import ndimage
import numpy as np
from skimage import filters, util, color, transform,exposure
from torch import dtype, int8
import pandas as pd

def img_manipulation(img, mode=0):
    # add manipulation
    if mode == 0:#添加噪声
        return util.random_noise(img) 
    elif mode == 1: 
        return exposure.rescale_intensity(1-filters.sobel(img)) 
    elif mode == 2:#旋转图像
        return transform.rotate(img, 45) 
    elif mode == 3:#改变对比度
        v_min, v_max = np.percentile(img, (0.2, 99.8))
        better_contrast = exposure.rescale_intensity(
            img, in_range=(v_min, v_max))
        return better_contrast 
    elif mode == 4:#改变伽马值
        adjusted_gamma_image = exposure.adjust_gamma(
            img, gamma=0.4, gain=0.9)
        return adjusted_gamma_image
    elif mode == 5:#模糊图像
        blured_image = ndimage.uniform_filter(img, size=(11, 11, 1))
        return blured_image
    elif mode == 6:
        return exposure.adjust_log(img)
    elif mode == 7:
        return exposure.adjust_sigmoid(img)


def getPatchesPerImage(filename, patch_size=256, stride=50):
    img = Image.open(filename)
    img_array = np.asarray(img)
    h, w = img_array.shape[:2]
    patches_list = []
    modes_list = []
    for idx_h in range(0, h - patch_size + 1, stride):
        for idx_w in range(0, w - patch_size + 1, stride):
            patch = img_array[idx_h:idx_h+patch_size, idx_w:idx_w+patch_size, :]
            mode = np.random.randint(0, 8, size=1)
            patch = img_manipulation(patch, mode)
            patches_list.append(patch)
            modes_list.append(mode)
    return patches_list, modes_list

def getAllImages(filedir, saveDir='./', patch_size=256, stride=50):
    dir_list = glob(filedir + '/*')
    print(len(dir_list))
    count = 0
# patches_list, modes_list = getPatchesPerImage(filename='./1.jpg')
# patches_list = np.array(patches_list, dtype='float32')
# patches_list = patches_list.reshape(patches_list.shape[0], -1)
# modes_list = np.array(modes_list, dtype='float32')
# modes_list = modes_list.reshape(modes_list.shape[0], -1)


getAllImages(
    filedir='F:\code\MachineLearning\datasets\sp-society-camera-model-identification\\train\\train')

# for idx, patch in enumerate(patches_list):
#     plt.imsave('./patches/%06d.jpg' % (idx), patch, format='png')
# img = Image.open('./1.jpg')
# img_array = np.asarray(img)
# gray = exposure.rescale_intensity(1 - filters.sobel(img_array))
# rgb = color.gray2rgb(gray)
# fig, axe = plt.subplots(1, 3)
# axe[0].imshow(gray)
# axe[1].imshow(rgb)
# axe[2].imshow(img_array)
# plt.imsave('./gray.jpg', gray)
# plt.show()
