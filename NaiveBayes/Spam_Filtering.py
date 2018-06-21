#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
#Note:正则表达式(regular expression)描述了一种字符串匹配的模式（pattern），
# 可以用来检查一个串是否含有某种子串、将匹配的子串替换或者从某个串中取出符合某个条件的子串等
import Bayes
import re
from numpy import *

#INPUT:bigString:词向量
#OUTPUT:长度大于2的单词(特征)
def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString) #r'\W'实际上是先re.compile('\W*')生成正则表达式对象:其中'\W*'表示匹配所有非数字和非字母,即所有非数字和非字母都会被split掉
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #只取长度大于2的单词,且将单词全变为小写

#词集模型的垃圾邮件分类测试
def spamTestOfvoc():
    docList =[];classList = [];fullText = []
    for i in range(1,26):#总共有50份文件,垃圾邮件25份,非垃圾邮件25份
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  #加入一个词向量样本到docList
        fullText.extend(wordList) #extend()方法使得fullText中的元素都是单个的单词(list类型),参考:https://www.cnblogs.com/tzuxung/p/5706245.html
        classList.append(1) #spam中的样本都是垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = Bayes.createVocabList(docList) #得到所有单词(特征)的词汇表
    trainingSet = range(50) #0-49
    testSet = []
    for i in range(10):#交叉验证,10个样本用于测试
        randIndex = int(random.uniform(0,len(trainingSet)))#生成一个在[0,len(trainingSet)的随机数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])#在训练集中去掉测试集
    trainMat =[];trainClasses = []
    for docIndex in trainingSet:#得到训练集和其对应的类别
        trainMat.append(Bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = Bayes.trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:#测试,得到错误率
        wordVector = Bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if Bayes.classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))

def spamTestOfbag():
    docList =[];classList = [];fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList) #extend()方法使得fullText中的元素都是单个的单词(list类型),参考:https://www.cnblogs.com/tzuxung/p/5706245.html
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = Bayes.createVocabList(docList)
    trainingSet = range(50);testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat =[];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(Bayes.bagOfWord2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = Bayes.trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = Bayes.bagOfWord2VecMN(vocabList,docList[docIndex])
        if Bayes.classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))

if '__main__':
    spamTestOfvoc()
    spamTestOfbag()