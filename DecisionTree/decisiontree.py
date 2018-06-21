#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from math import log
from numpy import *
import PlotTree
import pickle


#计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #返回的是数组的行数,len(dataSet[0,:])返回数组的列数,这里得到数据的总数量
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*math.log(prob,2)
    return shannonEnt

#数据处理,根据选择的特征属性和取值重新划分数据集,得到子数据集,以便计算信息熵
# INPUT:dataSet,特征的index,特征的值
# OUTPUT:除去特征i的subData
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1 :]) #extend()函数给List加入的是一个序列
            retDataSet.append(reducedFeatVec)
    return retDataSet

#Just for the test
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

# 通过信息增益得到最适合的节点特征
# 具体做法为: 得到每列的特征的取值范围,根据不同的取值得到subData,计算其信息熵,
# 再把所有子集的信息熵求和,最后求出每一列(特征)对应的信息增益,选取信息增益最大的作为最优特征
# INPUT:子集
# OUTPUT:最优特征在子集中的index
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEbtropy = calcShannonEnt(dataSet)
    bestInfoGain =0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #featList存储了dataSet的每一列的所有元素
        uniqueVals = set(featList) #set()函数创建一个无序不重复元素集合,这里就是为了得到每个特征有多少种取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value) #根据第i个特征的不同取值将dataSet分类,计算每一类的信息熵
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEbtropy - newEntropy #计算信息增益
        if (infoGain > bestInfoGain): #信息增益最大的即为最好的特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 当所有属性消耗完,所得的子集不是同一类,则采取投票的方式
# INPUT:最后叶节点的子集
# OUTPUT:最终的分类类别
def majorityCnt(classList):
    classCount ={}
    for vote in classList:#投票过程
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),reverse = True)#这里的key=operator.itemgetter(1)表示按照dict的value排序
    return sortedClassCount[0][0] #第一个key [0][1]则返回第一个value
#########
# INPUT:数据集,属性的名字(label)
# OUTPUT:决策树
# Note:这里通过递归调用的方式进行决策树的构建,递归的停止条件有两个:
# 1.所有的叶节点都是同一类
# 2.消耗完所有属性,通过最后的投票分类
############
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#如果是同一类,则停止
        return classList[0]
    if len(dataSet[0])==1:#如果属性消耗完了,则采用投票形式
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])#已经分类完的属性可以删除
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#存储决策树
def storeTree(inputTree,filename):
    fw = open(filename,'wb') #需要表明以二进制打开,否则会报错
    pickle.dump(inputTree,fw,0)
    fw.close()

#打开访问决策树
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

#打开数据集
def OpenData(filename):
    fr=open(filename)
    dataSet = [inst.strip().split('\t') for inst in fr.readlines()]

    return  dataSet

#进行测试
# INPUT:决策树,特征labels,测试向量
# OUTPUT:输出分类
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel = secondDict[key]
    return classLabel



if  '__main__':
    dataSet = OpenData('lenses.txt')
    Labels =  ['age','prescript','astigmatic','tearRate']
    test = ['young', 'myope', 'no', 'reduced']#no lense
    myTree = createTree(dataSet,Labels)#注意在构建决策数的过程中del了labels的成员,所以在测试的时候需要重新写出testLabels,而不能直接给labels
    PlotTree.createPlot(myTree)















