#!/usr/bin/env python
# coding=utf-8
# 对于图像识别而言，我们需要先进行图像数据的预处理
# 通过对于原始数据的读取，我们将图像转换成灰度图
# 通过灰度图像进行resize，做成标准的形式，转换成为[None,width,height,channel]
# 并且，图像中的数据和图像的类别标签是进行分开存储的

import os
import cv2
import numpy as np
import pickle

img_width = 40
img_height = 40

def convertImageData(parentDir,childDir,savePrefix):
    dataImg = []
    labelImg = []
    for dirpath,dirname,filenames in os.walk(parentDir+childDir):
        if len(dirname) == 0:
            n_class = int(dirpath.split("/")[-1])
            for filename in filenames:
                fileadspath = dirpath + "/" + filename
                rgb_img = cv2.imread(fileadspath)
                gray_img = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2GRAY)
                norm_img = cv2.resize(gray_img,(img_width, img_height),interpolation=cv2.INTER_CUBIC)
                norm_img.shape = img_width,img_height,1
                dataImg.append(norm_img)
                labelImg.append(n_class)
    
    info = np.array(dataImg,dtype = np.float32)
    handle_data = open(savePrefix + "_data.pkl",'wb+')
    pickle.dump(info,handle_data)
    handle_data.close()
    
    label_info = np.array(labelImg,dtype = np.float64)
    handle_label = open(savePrefix + "_label.pkl",'wb+')
    pickle.dump(label_info,handle_label)

    handle_label.close()

def DUCDATA():
    dataDir = "/opt/DataTest/"
    
    trainDir = "DUCD_train"
    testDir = "DUCD_test"

    convertImageData(dataDir, trainDir, "./DUC/DUC_train")
    convertImageData(dataDir, testDir, "./DUC/DUC_test")
    convertImageData(dataDir, testDir, "./DUC/DUC_valid")

def PAPERDATA():
    dataDir = "/opt/PaperBuilding/"

    convertImageData(dataDir,"train","./Paper/paper_train")
    convertImageData(dataDir,"test","./Paper/paper_test")
    convertImageData(dataDir,"valid","./Paper/paper_valid")

if __name__ == "__main__":
    PAPERDATA()
    #DUCDATA()
