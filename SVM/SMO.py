#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#随机选择i外的j
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#对拉格朗日乘子alpha进行裁剪
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj


# 简化版的smo算法
# INPUT:dataMatIn:输入训练数据,classLabels:分类标签,C:alpha阈值,即0<alpha<C,maxIter:最大迭代次数
#       toler:容错率,因为Ei是函数g(x)对xi的预测值和真实输出值yi之差(见<统计学习方法>Page127)
# OUTPUT:alhpas,b
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    b=0;m,n =shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter<maxIter):
        alphaPairsChanged = 0 #
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*\
                        (dataMatrix*dataMatrix[i,:].T))+b #f(x)<统计学习方法>Page124
            Ei = fXi - float(labelMat[i]) #<统计学习方法>Page127 (7.105)
            # 分类误差比toler大,需要继续迭代优化
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or \
                    ((labelMat[i]*Ei>toler) and \
                     (alphas[i]>0)):
                j = selectJrand(i,m) #选择alpha_2
                fXj = float(multiply(alphas,labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj-float(labelMat[j])
                alphaIold = alphas[i].copy()#alpha_1_old
                alphaJold = alphas[j].copy()#alpha_2_old
        #对alpha_2_new进行裁剪的上下界 (L,H)
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L==H:print("L==H");continue
                # <统计学习方法>Page128
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:print("eta>=0");continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta #这里的减号是因为eta和书中的正好取反了
                alphas[j] =clipAlpha(alphas[j],H,L)#alpha_2_new
                if (abs(alphas[j]-alphaJold)<0.00001):print("j not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j]) #alpha_1_new
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i]) and (C>alphas[i]):b=b1
                elif (0<alphas[j]) and (C>alphas[j]):b=b2
                else:b=(b1+b2)/2.0
                alphaPairsChanged += 1
                print("iter:%d i:%d,paris changed %d" % (iter,i,alphaPairsChanged))
        if(alphaPairsChanged == 0):iter+=1
        else:iter=0
        print("iteration number:%d"% iter)
    return b,alphas





if '__main__':
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)


