#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
from matplotlib import pyplot as plt

#示例数据集和标签
def loadSimpData():
    dataMat = matrix([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1.,1.],
                     [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#示例数据集可视化
def Plotfigure(dataMat,classLabels):
    x1=[];y1=[]
    x2=[];y2=[]
    m=len(classLabels)
    for i in range(m):
        if classLabels[i]==1:
            x1.append(dataMat[i,0])
            y1.append(dataMat[i,1])
        else:
            x2.append(dataMat[i,0])
            y2.append(dataMat[i,1])
    plt.plot(x1,y1,'ro',)
    plt.plot(x2,y2,'g*')
    plt.xlim([0,5])
    plt.show()

#特征分类函数,根据特征值是否大于threshVal分类到1和-1,threshIneq:左右树
#return:数据分类结果
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal] =-1.0 #所有dataMatrix第dimen列中小于threshVal的index处设置retArray[index]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0
    return retArray

#单层决策树的构建
# Note:这里的原理是:(1)对每个特征循环,选择的特征是最好的 (2)在特征的范围内选定步长,选择最合适的阈值
# (3)'lt'和'gt'分别代表两种情况:第一种:'lt':低于阈值的是分类为-1的,第二种:'gt':高于阈值的分类为-1  (因为并不知道超过阈值应该分为哪一类)
# (4)最终单层决策树的原则:minError,误差最小
# INPUT:dataArr:数据集 classLabels:数据集标签,D:数据集权重
# return:单层决策树,最小误差,分类结果
#得到的弱分类器不同,主要由于D不同,最后的分类误差权重也就不同
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump={};bestClasEst=mat(zeros(((m,1))))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()#rangeMin,rangeMax分别代表数据集中特征的最小和最大值
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        for  j in range(-1,int(numSteps)+1): #循环12次
            for inequal in ['lt','gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] =0
                weightedError = D.T*errArr #<统计学习方法>Page139 (8.8)或Page138 (8.1)
                print("split:dim %d,thresh %.2f,thresh ineqal:%s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] =i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#INPUT:dataArr:训练集 classLabels:训练集的标签  numIt:弱分类器最多的个数
#OUPUT:weakClassArr:弱分类器的线性组合
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#初始权重1/m
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#1/2*In((1-error)/error),分类器的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)#弱分类器的列表
        # print("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)#<统计学习方法>Page139(8.4)
        D = multiply(D,exp(expon))#
        D = D/D.sum()#下一个分类的各样本的权重D(i+1)
        aggClassEst += alpha*classEst # <统计学习方法>Page139(8.6)的f(x),即若分类器的线性组合
        # print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        # print("total error: ",errorRate,"\n")
        if errorRate == 0:break#两种情况停止:(1)40个弱分类器的组合 (2)分类误差为0
    return  weakClassArr,aggClassEst

#分类测试函数
def adaClassify(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)
if '__main__':
    dataMat,labelMat = loadSimpData()
    classifyArr,aggESt = adaBoostTrainDS(dataMat,labelMat)

    # print(adaClassify([0,0],classifyArr))



