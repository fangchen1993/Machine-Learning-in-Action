#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from os import listdir
from numpy import *
import Platt_SMO

######
# 将一个文件的数据变成1x1024的array
# 输入:要读的文件名
# 输出:处理后的数据
######
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr =trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' %(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages('digits/trainingDigits')
    b,alphas = Platt_SMO.smoP(dataArr,labelArr,200,0.001,10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = Platt_SMO.kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!= sign(labelArr[i]):errorCount +=1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('digits/testDigits')
    errorCount =0
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval = Platt_SMO.kernelTrans(sVs,dataMat[i,:],kTup)
        predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict) != sign(labelArr[i]):errorCount +=1
    print("the test error rate is: %f" % (float(errorCount)/m))

if '__main__':
    testDigits()