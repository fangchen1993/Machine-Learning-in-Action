#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  //
# @Author  : FC
# @Site    : 2655463370@qq.com
# @license : BSD
from numpy import *
import SMO
#内层:alphaJ
#外层:alphaI

#定义数据结构体:
    # dataMatIn:训练数据
    # classLabels:数据标签
    # C:alpha阈值
    # toler:容错率
    # kTup:核函数类型和参数,'rbf'是高斯径向基函数(radial basis function)  kTup=('rbf',sigma)
    # eCache:误差E的缓存区
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0] #返回dataMatIn的行数
        self.alphas = mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)#按列插值

def calcEk(oS,k):
    # fXK = float(multiply(oS.alphas,oS.labelMat).T*\
    #             (oS.X*oS.X[k,:].T))+oS.b
    # Ek = fXK - float(oS.labelMat[k])
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)#<统计学习方法>Page127g(x)
    Ek = fXk - float(oS.labelMat[k])#预测值和真实值的误差
    return Ek

#alphaJ的选择:第一次循环随机抽取,后面的循环满足条件max|Ei-Ej|选取
#eCache:缓存误差E的值
def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei] #eCahe是所有误差的缓存区
    vaildEcacheList = nonzero(oS.eCache[:,0].A)[0]
    #.A表示转化为array,注意nonzero返回的成对的值,这里validEcache返回的是第一列非零的行索引.见博客:
    if(len(vaildEcacheList))>1:
        for k in vaildEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k) #计算Ek,有索引k即可
            deltaE = abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:#随机选择alphaJ
        j=SMO.selectJrand(i,oS.m)
        Ej =calcEk(oS,j)
    return j,Ej

#更新误差缓存区eCache,一旦更新了设置标志位为1
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

#内循环
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or \
            ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej = selectJ(i,oS,Ei)#选择alphaJ
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        #对最有值的边界确定
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j]-oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print("L==H");return 0
        #eta = 2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T- oS.X[j,:]*oS.X[j,:].T
        eta = 2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]#<统计学习方法>Page128
        if eta >=0:print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = SMO.clipAlpha(oS.alphas[j],H,L)#将alphaJ限定在[L,H]之间 更新alphas[j]
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough!");return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])#更新alpha[i]
        updateEk(oS,i)
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        # b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*\
        #    (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        # b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*\
        #    (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):oS.b=b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0
        return 1
    else:return 0



def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    while(iter<maxIter) and ((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):#小样本不检查违背KKT条件,全部更新?
                alphaPairsChanged += innerL(i,oS)
            print("fullSet,iter: %d i:%d,pairs changed %d"%(iter,i,alphaPairsChanged))
            iter +=1
        else:#第二轮更新,之针对间隔内的alpha进行迭代更新
            nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound,iter:%d i:%d,pairs changed %d"%(iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:entireSet = False
        elif (alphaPairsChanged == 0):entireSet=True
        print("iteration number:%d"%iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)#<统计学习方法>Page111 (7.50)
    return w

def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':K=X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:raise NameError('HOUSTON We Have a Problem -- That Kernel is not recognized')
    return K

def testRbf(k1=1.3):
    dataArr,labelArr = SMO.loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelLSV = labelMat[svInd]
    print("there are %d Support Vectors"% shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelLSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):errorCount +=1
    print("the training error rate is: %f"% (float(errorCount)/m))
    dataArr,labelArr = SMO.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T*multiply(labelLSV,alphas[svInd])+b
        if sign(predict)!= sign(labelArr[i]):errorCount +=1
    print("the test error rate is: %f"% (float(errorCount)/m))



if '__main__':
    testRbf()
