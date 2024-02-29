# -*- coding:utf-8 -*-

"""
@author:gz
@file:featureExtract.py.py
@time:2021/7/2014:51
"""
from PIL import Image, ImageOps
from PIL.ImageFilter import SMOOTH_MORE
import numpy as np
import glob
import cv2
from skimage.feature import hessian_matrix
from skimage.feature import hessian_matrix_eigvals
from skimage import exposure
from skimage import filters,io,color
import torch
import os
from PIL.ImageFilter import CONTOUR, EMBOSS, SMOOTH_MORE

def myHessian(img,sigma,thr):
    h_elems = hessian_matrix(img, sigma, order='rc')
    h_elems[0] = (sigma ** 2) * h_elems[0]
    h_elems[1] = (sigma ** 2) * h_elems[1]
    h_elems[2] = (sigma ** 2) * h_elems[2]

    (lambda1, lambda2) = hessian_matrix_eigvals(h_elems)
    lambda2[lambda2 == 0] = 1e-10
    rb = lambda1/lambda2
    s2 = lambda1 ** 2 + lambda2 ** 2
    beta1 = 0.5 #########beta
    c = np.max(img)/4 #########c
    c = 0.05

    filtered = np.exp(-(rb**2) / (2*beta1**2)) * (np.ones(np.shape(img)) -
                                      np.exp(-s2 / (2 * c **2)))
    filtered[lambda2>0]=0
    filtered = torch.sigmoid(torch.tensor(filtered-np.ones(np.shape(img))/2))
    filtered = filtered.detach().numpy()
    filtered[filtered < thr] = 0
    return filtered


#1 HE
def HE(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('/') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img1 = Image.fromarray(img)
        img2 = img1.filter(SMOOTH_MORE)
        img=np.array(img2)
        Image.fromarray(img).convert('L').save(os.path.join(outPath ,imgnames))


# 2 CLAHE
def CLAHE(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        print(np.max(cl1))
        Image.fromarray(cl1).convert('L').save(os.path.join(outPath ,imgnames))


# 3 Hessian
def Hessian(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('/') + 1:]
        myimg = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = myHessian(myimg, 3, 0.45)
        newimg = myimg * img
        Image.fromarray(newimg).convert('L').save(os.path.join(outPath , imgnames))

# 4 Bilateral
def Bilateral(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))

# 5 Gaussian
def Gaussian(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = cv2.GaussianBlur(img,(5,5),0)
        Image.fromarray(img).convert('L').save(os.path.join(outPath ,imgnames))

# 6 median
def Median(inPath, outPath): 
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = cv2.medianBlur(img, 3)
        Image.fromarray(img).convert('L').save(os.path.join(outPath ,imgnames))

# 7 Moment
def Moment(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)

        moments = cv2.moments(img)  # 
        humoments = cv2.HuMoments(moments)  
        humoment = (np.log(np.abs(humoments))) / np.log(10)
        Image.fromarray(humoment).convert('L').save(os.path.join(outPath ,imgnames))

# 8 Gamma
def Gamma(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = exposure.adjust_gamma(img, 2)
        Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))

# 9 Laplacian
def Laplacian(inPath, outPath): 
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img1 = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
        img1 = 255*(img1 - np.min(img1))/(np.max(img1) - np.min(img1))
        img = img - img1
        Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))

# 10 Sobel 
def Sobel(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        img = 255*(img - np.min(img))/(np.max(img) - np.min(img))
        Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))

# 11  Canny 
def Canny(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('\\') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        img = cv2.Canny(img,50,100)
        Image.fromarray(img).convert('L').save(os.path.join(outPath, imgnames))

# 12 Gabor 
def Gabor(inPath, outPath):
    imgs = glob.glob(inPath)
    for imgname in imgs:
        imgnames = imgname[imgname.rindex('/') + 1:]
        img = cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), -1)
        H, W = img.shape
        out = np.zeros([H, W], dtype=np.float32)
        theta = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        for i, t in enumerate(theta):
            # gabor filtering
            out1 = cv2.getGaborKernel(ksize=(5, 5), sigma=5, theta=t, lambd=10, gamma=1.2)
            result = cv2.filter2D(img, -1, out1)
            # add gabor filtered image
            out += result
        out = out / out.max() * 255
        out = out.astype(np.uint8)
        Image.fromarray(out).convert('L').save(os.path.join(outPath, imgnames))