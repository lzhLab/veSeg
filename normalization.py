from PIL import Image, ImageOps
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix
from skimage.feature import hessian_matrix_eigvals
from skimage import exposure
from skimage import filters,io,color
import torch
import os
inPath = 'dataset/img/*'
outPath = 'dataset/'
data = []

imgs = glob.glob(inPath)
for imgname in imgs:
    imgnames = imgname[imgname.rindex('/') + 1:]
    img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    img1 = img.flatten()
    data.append(img1)

data = np.hstack(data)

del_arr = np.delete(data, np.where(data == [0.0]), axis=0)
arr_mean = np.mean(del_arr)
arr_std = np.std(del_arr, ddof=1)
data1 = np.clip(data, arr_mean-3*arr_std, arr_mean+3*arr_std)

for imgname in imgs:
    imgnames = imgname[imgname.rindex('/') + 1:]
    img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)

    img = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    img = np.clip(img, arr_mean - 3 * arr_std, arr_mean + 3 * arr_std)
    i = 3 * arr_std
    j = 6 * arr_std
    img = 255*(img + i ) / j
    Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))