#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
import  LogRegres
from numpy import *

def classifyVector(inX,weights):
    prob = LogRegres.sigmod(sum(inX*weights))
    if prob > 0.5 : return 1.0
    else: return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights,params = LogRegres.stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))

        if int(classifyVector(array(lineArr),trainWeights))!= int(currLine[-1]):
            errorCount += 1

    errorRate = errorCount/numTestVec
    print("the error rate of this is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" %(numTests,errorSum/numTests))

if '__main__':
    multiTest()