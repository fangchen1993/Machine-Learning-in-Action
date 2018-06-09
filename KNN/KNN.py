#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
#####################
#function:KNN算法实现
# 输入:
# inX:待分类的样本
# dataSet:训练样本
# labels:对应训练样本的标签
# k:KNN算法的参数,选择近邻的数目
# 输出:
# 待分类样本的类别
####################
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet #tile(a,(m,n)):将a元素复制成(m,n)的array
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #每一行的误差的求和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#argsort()函数将误差进行从小到大排序,并返回其在原数组中的index,主要是为了对应label
    classCount = {} #字典,存储k个近邻中出现的类别和其出现次数
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1 #get(voteLabel,0),若voteLabel存在,返回其对应的值,若不存在则返回指定的0(新的类别,第一次出现)
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),reverse=True)
    #iteritems()返回字典的指针(即一组一组的访问字典),其中的key=operator.itemgetter(1)表示排序的根据是dict的value,而不是dict的key
    return sortedClassCount[0][0]
#######################
# function:处理训练样本文件
# 输入:
# 训练样本集文件名
# 输出:
# 训练样本和对应的label
#########################
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #移除开头和结尾的空格
        listFromLine = line.split('\t')#根据空格将字符串分割开来
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


###############
# 输入:训练样本dataSet
# 输出:归一化之后的样本 normDataSet
###############
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 算法的测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],4)
        print("the classifier came back with:%d,the real answer is:%d"\
              %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount += 1.0
    print("the total error rate is %f"%(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all','in small does','in large does']
    percentTats = float(input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(input(\
        "frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datLabels,4)
    print("You will probably like this person: ",\
          resultList[classifierResult-1])

if '__mian__':
    groups,labels = file2matrix('datingTestSet2.txt')
    print(groups)


