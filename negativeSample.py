import glob
from math import exp

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def Move(delta_x, delta_y):  # shift 
    # delta_x>0: move left，delta_x<0: move right
    # delta_y>0: move up，delta_y<0: move down
    return np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

img = r"dataset/img/*"
# mask = r"dataset\mask\*"
f1_path = r"dataset/img"
f2_path = r"dataset/clahe"
f3_path = r"dataset/gabor"
f4_path = r"dataset/gamma"
f5_path = r"dataset/gauss"
f6_path = r"dataset/hessian"
f7_path = r"dataset/laplacian"
f8_path = r"dataset/media"
f9_path = r"dataset/sb"
f10_path = r"dataset/sobel"
f11_path = r"dataset/SMOOTH_MORE"
f12_path = r"dataset/canny"
f13_path = r"dataset/MeanFilter"
f14_path = r"dataset/scharr"
f15_path = r"dataset/blur"
f16_path = r"dataset/CONTOUR"
f17_path = r"dataset/DETAIL"
f18_path = r"dataset/EDGE_ENHANCE"
f19_path = r"dataset/EDGE_ENHANCE_MORE"
f20_path = r"dataset/EMBOSS"
f21_path = r"dataset/FIND_EDGES"
f22_path = r"dataset/minfilter"
f23_path = r"dataset/SHARPEN"
f24_path = r"dataset/SMOOTH"
f25_path = r"dataset/y"


file_obj = open(r"dataset/xiugaibiankuang13.txt","w")

mask = glob.glob(img)
# select_imgs = []
for imgname in mask:
    imgnames = imgname[imgname.rindex('/') + 1:]
    mymask = cv2.imdecode(np.fromfile(imgname.replace('img','mask'), dtype=np.uint8), -1)
    # vessel mask is 1
    # xg = np.int64(mymask>0)
    xg = mymask>0

    if np.sum(xg)>1500:
        f1 = cv2.imdecode(np.fromfile(os.path.join(f1_path, imgnames), dtype=np.uint8), -1)
        f2 = cv2.imdecode(np.fromfile(os.path.join(f2_path, imgnames), dtype=np.uint8), -1)
        f3 = cv2.imdecode(np.fromfile(os.path.join(f3_path, imgnames), dtype=np.uint8), -1)
        f4 = cv2.imdecode(np.fromfile(os.path.join(f4_path, imgnames), dtype=np.uint8), -1)
        f5 = cv2.imdecode(np.fromfile(os.path.join(f5_path, imgnames), dtype=np.uint8), -1)
        f6 = cv2.imdecode(np.fromfile(os.path.join(f6_path, imgnames), dtype=np.uint8), -1)
        f7 = cv2.imdecode(np.fromfile(os.path.join(f7_path, imgnames), dtype=np.uint8), -1)
        f8 = cv2.imdecode(np.fromfile(os.path.join(f8_path, imgnames), dtype=np.uint8), -1)
        f9 = cv2.imdecode(np.fromfile(os.path.join(f9_path, imgnames), dtype=np.uint8), -1)
        f10 = cv2.imdecode(np.fromfile(os.path.join(f10_path, imgnames), dtype=np.uint8), -1)
        switch = {'f1': f1,  
                  'f2': f2,
                  'f3': f3,
                  'f4': f4,
                  'f5': f5,
                  'f6': f6,
                  'f7': f7,
                  'f8': f8,
                  'f9': f9,
                  'f10': f10,
                  }

        # through shift to select non-vessel masks with lable 2
        M = np.float32([[1, 0, 15], [0, 1, 15]])
        M1 = np.float32([[1, 0, -15], [0, 1, 15]])
        M2 = np.float32([[1, 0, 15], [0, 1, -15]])
        M3 = np.float32([[1, 0, -15], [0, 1, -15]])

        dst = cv2.warpAffine(mymask, M, (512, 512))  
        dst1 = cv2.warpAffine(mymask, M1, (512, 512))
        dst2 = cv2.warpAffine(mymask, M2, (512, 512))
        dst3 = cv2.warpAffine(mymask, M3, (512, 512))

        fxg = (dst > 0) * 2
        fxg1 = (dst1 > 0) * 2
        fxg2 = (dst2 > 0) * 2
        fxg3 = (dst3 > 0) * 2

        # selection
        xg_fxg = xg + fxg
        xg_fxg1 = xg + fxg1
        xg_fxg2 = xg + fxg2
        xg_fxg3 = xg + fxg3
        xg_fxg[xg_fxg > 2] = 1
        xg_fxg1[xg_fxg1 > 2] = 1
        xg_fxg2[xg_fxg2 > 2] = 1
        xg_fxg3[xg_fxg3 > 2] = 1

        label1 = np.argwhere(xg_fxg == 1)
        for it in label1:
            x, y = it

            train_str = "1 "

            for n in range(10):
                    f1 = switch["f" + str(n + 1)]
                    c = 0
                    for i in range(3):
                        for j in range(3):

                            if f1[x + i - 1, y + j - 1] != 0:
                               train_str = train_str + str(c + 9 * n) + ":" + "%.5f" % (f1[x + i - 1, y + j - 1] / 255) + " "
                            c = c + 1
            file_obj.writelines("\n"+ train_str)
        label2 = np.argwhere(xg_fxg == 2)
        for it in label2:
            x, y = it
            train_str = "0 "
            for n in range(10):
                   f = switch["f" + str(n + 1)]
                   c = 0
                   for i in range(3):
                       for j in range(3):

                           if f[x + i - 1, y + j - 1] != 0:
                               train_str = train_str + str(c + 9 * n) + ":" + "%.5f" % (f[x + i - 1, y + j - 1] / 255) + " "
                           c = c + 1

            file_obj.writelines("\n"+ train_str)
        label3 = np.argwhere(xg_fxg1 == 2)
        for it in label3:
            x, y = it
            train_str = "0 "

            for n in range(10):
                f1 = switch["f" + str(n + 1)]
                c = 0
                for i in range(3):
                    for j in range(3):

                        if f1[x + i - 1, y + j - 1] != 0:
                            train_str = train_str + str(c + 9 * n) + ":" + "%.5f" % (
                                    f1[x + i - 1, y + j - 1] / 255) + " "
                        c = c + 1
            file_obj.writelines("\n" + train_str)
        label4 = np.argwhere(xg_fxg2 == 2)
        for it in label4:
            x, y = it
            train_str = "0 "

            for n in range(10):
                f1 = switch["f" + str(n + 1)]
                c = 0
                for i in range(3):
                    for j in range(3):

                        if f1[x + i - 1, y + j - 1] != 0:
                            train_str = train_str + str(c + 9 * n) + ":" + "%.5f" % (
                                    f1[x + i - 1, y + j - 1] / 255) + " "
                        c = c + 1
            file_obj.writelines("\n" + train_str)
        label5 = np.argwhere(xg_fxg3 == 2)
        for it in label5:
            x, y = it
            train_str = "0 "

            for n in range(10):
                f1 = switch["f" + str(n + 1)]
                c = 0
                for i in range(3):
                    for j in range(3):

                        if f1[x + i - 1, y + j - 1] != 0:
                            train_str = train_str + str(c + 9 * n) + ":" + "%.5f" % (
                                    f1[x + i - 1, y + j - 1] / 255) + " "
                        c = c + 1
            file_obj.writelines("\n" + train_str)
file_obj.close()