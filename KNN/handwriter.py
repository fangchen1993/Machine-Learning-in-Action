#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
import os
import operator

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

#手写数字识别主程序
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')#listdir()函数返回文件夹下各文件的名字
    m = len(trainingFileList) #文件数
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i] #文件读取,得到的是文件的名字,如'6_179.txt'
        fileStr = fileNameStr.split('.')[0] #这里是得到'6_179'
        classNumStr = int(fileStr.split('_')[0])#这里得到'6',也就是我们的标签
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    #####
    #下面的代码用于测试
    #####
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat,hwLabels,4)
        print("the classifier came back with:%d,the real answer is : %d"\
              % (classifierResult,classNumStr))
        if (classifierResult != classNumStr):errorCount +=1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n total error rate is %f" % (errorCount/float(mTest)))



if '__main__':
    handwritingClassTest()