# -*- coding:utf-8 -*-

"""
@author:gz
@file:featureCombination.py
@time:2021/7/228:10
"""

import glob
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import threading
import time
from alive_progress import alive_it

# img = r"dataset/img/*"
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

path = [f1_path, f2_path, f3_path, f4_path, f5_path, f6_path, f7_path, f8_path, 
        f9_path, f10_path, f11_path, f12_path, f13_path, f14_path, f15_path, f16_path, 
        f17_path, f18_path, f19_path, f20_path, f21_path, f22_path, f23_path, f24_path, f25_path]


img = r"dataset/img/*"
source_dir = 'img'
target_dir = 'rst'
num_continuous_img = 7 
num_thread = 31 # number of thread
num_conv = 7 # kernal size

class myThread(threading.Thread):
    
    def __init__(self, threadID, name , path_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path_list = path_list

    def run(self):
        for group in alive_it(self.path_list):

            temp = num_continuous_img // 2
            temp1 = num_conv // 2
            middle = group[temp]
            if num_continuous_img == 1:
                file_path = middle.replace(source_dir, target_dir).replace('.png', '.txt')
            else:
                file_path = middle.replace(source_dir, target_dir).replace('_%d.png' % temp, '.txt')
            file_obj = open(file_path,"w")

            switchs = []
            for g in group:
                img_name = g[g.rindex('/') + 1:]
                switch = []
                for p in path:
                    img = cv2.imdecode(np.fromfile(os.path.join(p, img_name), dtype=np.uint8), -1)
                    switch.append(img)
                switchs.append(switch)

            img = cv2.imdecode(np.fromfile(middle, dtype=np.uint8), -1)

            for it in np.argwhere(img>=0):
                x, y = it

                if(x >= temp1 and y >= temp1 and x <= (512 - temp1 - 1) and y <= (512 - temp1 - 1)):

                    train_str = "0 "
                    for s, switch in enumerate(switchs):
                        for n in range(25):
                            f = switch[n]
                            c = 0

                            for i in range(num_conv):
                                for j in range(num_conv):

                                    if f[x + i - temp1, y + j - temp1] != 0:
                                        train_str = train_str + str(c + (num_conv**2) * n + 25 * (num_conv**2) * s) + ":" + "%.5f" % ( f[x + i - temp1, y + j - temp1] / 255) + " "
                                    c = c + 1
                    file_obj.writelines("\n" + train_str)

            file_obj.close()

def featureCombination(img):
    img_paths = glob.glob(img)
    img_group = []
    group = []

    for o in img_paths:
        num = o.split('_')[-1].split('.')[0]
        if num_continuous_img == 1:
            group.append(o)
            img_group.append(group)
            group = []
            continue
        if num != str(num_continuous_img - 1):
            group.append(o)
        else:
            group.append(o)
            img_group.append(group)
            group = []
            
    split_group = []
    temp = ((len(img_group) // num_thread) + 1) 
    for x in range(num_thread):
        if x * temp < len(img_group):
            if (x + 1) * temp >= len(img_group):
                split_group.append(img_group[x * temp : len(img_group)])
                break
            else:
                split_group.append(img_group[x * temp : (x + 1) * temp])

    threads = []
    for i,group in enumerate(split_group):
        thread_No = i
        temp_thread = myThread(thread_No, 'thread-%d' % thread_No, group)
        temp_thread.start()
        threads.append(temp_thread)

    for t in threads:
        t.join()