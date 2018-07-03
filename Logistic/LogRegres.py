#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD

from numpy import *
from matplotlib import pyplot as plt

# 文件读取,并将data和label分开
# OUTPUT:dataMat:训练集;lableMat:训练集对应的标签
# Note:这里默认X0为1.0
def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return array(dataMat),array(labelMat)
#sigmod函数
def sigmod(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)

    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmod(dataMatrix*weights)
        error = (labelMat - h)
        weights += alpha*dataMatrix.transpose()*error
    return weights

#数据划分图形化
def plotBestFit(wei):
    # weights = wei.getA() #将mat类型的wei转成array
    dataMat ,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2] #Z=w0X0+w1X1+w2X2,令Z=0,得到X1=-(w0X0+w1X1)/w2 此处X0=1
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,calssLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i]*weights))
        error = calssLabels[i] - h
        weights = weights + alpha*error*dataMatrix[i]

    return  weights

#随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    params =[]
    weights = ones(n)
    params.append(weights)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex))) #随机的选取起点
            h = sigmod(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
        params.append(weights)
    return weights,params

#迭代次数和回归系数的关系图
def PlotParams(numbers,params):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    x1 = [];y1=[]
    for i in range(numbers):
        x1.append(i);y1.append(params[i][0])
    ax1.plot(x1,y1)
    plt.xlabel('Iterations')
    plt.ylabel('X0')

    ax2 = fig.add_subplot(312)
    x2 = [];y2=[]
    for i in range(numbers):
        x2.append(i);y2.append(params[i][1])
    ax2.plot(x2,y2,'r')
    plt.xlabel('Iterations')
    plt.ylabel('X1')

    ax1 = fig.add_subplot(313)
    x3 = [];y3=[]
    for i in range(numbers):
        x3.append(i);y3.append(params[i][2])
    ax1.plot(x3,y3,'y')
    plt.xlabel('Iterations')
    plt.ylabel('X2')
    plt.show()


# if '__main__':
#     dataMat,labelMat = loadDataSet()
#     weights,params = stocGradAscent1(dataMat,labelMat,1000)
#     plotBestFit(weights)
if '__main__':
    dataMat,labelMat = loadDataSet()
    numberIters = 4000
    weights,params = stocGradAscent1(dataMat,labelMat,numberIters)
    PlotParams(numberIters,params)

