#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
import Adaboost
import matplotlib.pyplot as plt
from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#ROC曲线
def plotROC(predStrengths,classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)#阳性样本个数
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedINdicies = predStrengths.argsort()#将predStrength从小到达排列,取index,即分类得出的几率的排序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedINdicies.tolist()[0]:#sortedINdicies.tolist()[0]:返回soredINdixies的一维列表
        if classLabels[index] == 1.0:
            delX = 0;delY = yStep
        else:
            delX = xStep;delY=0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],'r*')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    # print("the Area Under the Curve is: ",ySum*xStep)


if '__main__':
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifyArray,aggClassEst = Adaboost.adaBoostTrainDS(dataArr,labelArr,1000)#aggClassEst:f(x)在所有样本的分类结果,未经过sign函数,其len取决于样本个数
    # print(aggClassEst)

    plotROC(aggClassEst.T,labelArr)


