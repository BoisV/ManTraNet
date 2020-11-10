from numpy.core.defchararray import count, mod
from numpy.core.fromnumeric import clip
from numpy.core.numeric import indices
from skimage.morphology import disk
from matplotlib import pyplot as plt
from skimage.morphology.selem import star
from skimage.util.dtype import convert
import torch
import numpy as np
from PIL import Image
from glob import glob
from scipy import ndimage
from skimage import filters, util, transform, exposure
from torch import var
from torch.utils.data import Dataset
from scipy import signal
import os

def img_manipulation(img, mode=0):
    # add manipulation
    if mode == 0:  # gaussian噪声
        return util.random_noise(img, mode='gaussian', mean=0.5, var=0.3)
    elif mode == 1:  # salt噪声
        return util.random_noise(img, mode='salt')
    elif mode == 2:  # 直方图均衡化
        img = np.transpose(img, axes=[2, 0, 1])
        img1 = exposure.equalize_hist(img[0])
        img2 = exposure.equalize_hist(img[1])
        img3 = exposure.equalize_hist(img[2])
        img = np.stack([img1, img2, img3], axis=0)
        return np.transpose(img, [1, 2, 0])
    elif mode == 3:  # 改变对比度
        return exposure.rescale_intensity(1 - filters.sobel(img))
    elif mode == 4:  # 改变伽马值
        adjusted_gamma_image = exposure.adjust_gamma(
            img, gamma=0.4, gain=0.9)
        return adjusted_gamma_image
    elif mode == 5:  # 扭曲
        img = transform.swirl(
            img, center=[100, 100], rotation=0.3, strength=10, radius=120)
        return img
    elif mode == 6:  # 改变尺寸
        return transform.resize(img, (img.shape[0]*1.5, img.shape[1]*2))
    elif mode == 7:  # 反相
        return util.invert(img)


def convertOneImg2Patches(filename, patch_size=256, number=40):
    img = Image.open(filename)
    img_array = np.asarray(img)
    h, w = img_array.shape[:2]
    patches_list = []
    for _ in range(number):
        y = np.random.randint(0, h-patch_size+1)
        x = np.random.randint(0, w-patch_size+1)
        patch = img_array[y:y+patch_size, x:x+patch_size, :]
        patches_list.append(patch)
    return patches_list


def convertImgInDir2Patches(filedir, saveDir='./dataset', patch_size=256, number=100000, train=0.8,val = 0.1, test=0.1):
    dir_list = glob(filedir + '/*')
    numberPerImg = number // (275 * len(dir_list))
    if numberPerImg == 0:
        numberPerImg = 1
    patches_list = []
    for dir in dir_list:
        print('start reading %s...' % (dir))
        for img in glob(dir+'/*'):
            X = convertOneImg2Patches(img, patch_size=256, number=numberPerImg)
            patches_list.extend(X)
        print('finish reading %s...' % (dir))
        if len(patches_list) >= number:
            break
    print('get all patches succeed!')
    indices = range(len(patches_list))
    np.random.shuffle(indices)
    if train + val + test > 1:
        assert('sum of train, val and test must be 1')
    patches_list = patches_list[indices[:number]]
    patches_list_train = patches_list[:train*number]
    patches_list_train = patches_list[train*number: (train+val)*number]
    patches_list_test = patches_list[(train+val)*number:]
    for mode in range(0, 8):
        count = 0
        batch = train * number // 8
        for patch in patches_list_train[mode*batch:(mode+1)*batch]:
            p = img_manipulation(patch, mode)
            count += 1
            plt.imsave(os.path.join(saveDir, 'train/mode%d/patches/%5d.png' % (mode, count)), p, format='png')
        
        count = 0
        batch = val * number // 8
        for patch in patches_list_train[mode*batch:(mode+1)*batch]:
            p = img_manipulation(patch, mode)
            count += 1
            plt.imsave(os.path.join(
                saveDir, 'val/mode%d/patches/%5d.png' % (mode, count)), p, format='png')

        count = 0
        batch = test * number // 8
        for patch in patches_list_test[mode*batch:(mode+1)*batch]:
            p = img_manipulation(patch, mode)
            count += 1
            plt.imsave(os.path.join(saveDir, 'test/mode%d/patches/%5d.png' %
                                    (mode, count)), p, format='png')



def loadDataset(filedir):
    pass


# class KCMIDataset(Dataset):
#     def __init__(self) -> None:

if __name__ == "__main__":
    convertImgInDir2Patches(
        filedir='F:\code\MachineLearning\datasets\sp-society-camera-model-identification\\train\\train', saveDir='./dataset', number=100,)
