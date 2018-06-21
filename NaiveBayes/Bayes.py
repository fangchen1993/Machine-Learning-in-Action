#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *

#创建6个实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#创建文档中不重复出现的词的列表,返回一个list
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

# INPUT:词汇列表vocabList,输入词向量inputSet
# OUTPUT:inputSet中每个单词在vocabList中的位置,输出列表中'1'表示单词的位置
#词集模型
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)#[0,0,...]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in my Vocabulary!" % word) #vocabulary:词汇
    return  returnVec

#INPUT:trainMatrix:所有样本的词汇集合,即是每个样本通过setOfWords2Vec后得到的集合 trainCategory:类别列表
#OUTPUT:p0Vect p1Vect pAbusive:类别0的条件概率p(w|c=0),类别1的条件概率p(w|c=1),类别概率p(c)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#样本个数,词向量个数
    numWords = len(trainMatrix[0]) #词汇表长度,即特征总个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0 ; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #统计类别1中每个单词(特征)出现的次数
            p1Denom += sum(trainMatrix[i])#p1Denom:类别1中单词的总数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom) #类别1中每个单词(特征)的出现概率
    p0Vect = log(p0Num/p0Denom) #类别0中每个单词(特征)出现的概率
    return p0Vect,p1Vect,pAbusive

#INPUT:vec2Classify:输入词向量的word2vec模型 p0Vec,p1Vec,pClass1:p0Vect,p1Vect,pAbusive
#OUTPUT:输入词向量的类别
#Note:这里用log求和实际上进行合并,log(f),f是一个求乘积的运算,展开后是贝叶斯条件概率公式,分母p(w)对于分类来说是没有影响的
#同理,整个的概率log后也是对分类没影响的,因为p1和p0的相对大小不会变
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

#简单的分类示例
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#将word2vec(词集模型)改为word2vecMN(词袋模型),以应对多次出现的单词
def bagOfWord2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec



