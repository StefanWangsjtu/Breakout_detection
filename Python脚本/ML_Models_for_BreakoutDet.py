# 
#                  This Python File is built for breakout detection in fast EDM drilling.


import numpy as np
import sys
from os import listdir # 返回文件夹下包含的文件名列表
from sklearn import svm
import pickle
import os
import time
from sklearn.preprocessing import StandardScaler

sys.path.append("/home/stefan/PythonFiles/dataFiles/")

path = "/home/stefan/PythonFiles/dataFiles/"

# 用于保存model文件生成时间
currTime = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

# 此函数用于导入测试集数据,返回列表类型的dataSet和labelSet
def loadTestDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# 归一化处理
def normalLization(dataSet):
    dataMat = np.array(dataSet)
    mean = dataMat.mean(axis=0)
    std = np.std(dataMat, axis=0)
    dataMat = dataMat - mean
    dataMat = dataMat / std
    return dataMat

# 特征缩放
def featScaling(dataSet):
    dataMat = np.array(dataSet)
    maxLine = np.max(dataMat, axis=0)
    #minLine = np.min(dataMat, axis=0)
    #rangeLine = maxLine - minLine
    dataMat = dataMat / maxLine
    return dataMat

# 选择模型
Models = dict(SVM = 1, AdaBoost = 2, KNN = 3, Decision_Tree = 4, Logistic_Regression = 5)
ModelSelction = Models["SVM"]

# **********************************************************************************************************************
#                    ################         Support Vector Machine       ##################
# **********************************************************************************************************************

# 此函数用于导入数据,返回列表类型的dataSet和labelSet
def loadSVMDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# 用于存放支持向量机的相关信息
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 存放偏差
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 把所有的可能核函数变换后的两向量值放在矩阵K中
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

# 计算误差，oS是个类对象
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T*oS.K[:, k]+oS.b)
    Ek = fXk-float(oS.labelMat[k])
    return Ek

# 随机选择一个数
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

# 保证aj在H和L之间
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 选择另一个需要更新的alphaJ，与i有最大的偏差
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 将Ei置为有效，即第一个标志位为1.
    oS.eCache[i] = [1, Ei]
    # .A返回一个数组，np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 此函数用于更新偏差
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# 此函数用于进行核变换，仅支持线性核和rbf核
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :]-A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("The Kernel doesn't exist!")

    return K

# 内循环
def innerL(i, oS):
    # 计算第i个样本预测值与真值之间的误差
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei < -oS.tol)and(oS.alphas[i] < oS.C))\
            or((oS.labelMat[i]*Ei > oS.tol)and(oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j]-oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        eta = 2.0*oS.K[i, j]-oS.K[i, i]-oS.K[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        # if(abs(oS.alphas[j]-alphaJold) < 0.00001):
            # print("j is not moving enough")
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        # 更新存放在eCache中i对应的偏差
        updateEk(oS, i)
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i]\
             - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j]\
             - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if(0 < oS.alphas[i])and(oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 < oS.alphas[j])and(oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).T, C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter)and((alphaPairsChanged > 0)or(entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            # print("fullSet,iter:%d,i:%d,pairs changed %d"%(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound.iter:%d,i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number:%d" % iter)
    return oS.b, oS.alphas

# 计算w
def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr)
    labelMat=np.mat(classLabels).T
    m,n=np.shape(X)
    w=np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
    return w

# 训练模型，得到weight和bayes
def StefanSVM(trainingSet, trainingLabel, indexList):
    # bayes, alpha = smoP(trainingSet, trainingLabel, C=indexList[0], toler=indexList[1], maxIter=indexList[2])
    #weights = calcWs(alpha, trainingSet, trainingLabel)
    trainingSet = np.array(trainingSet)
    trainingLabel = np.array(trainingLabel).flatten()
    SVM_Classifier = svm.SVC(C=0.6, kernel='rbf',decision_function_shape='ovo')
    # SVM_Classifier = svm.SVC(C=5, kernel='linear',  decision_function_shape='ovo')
    SVM_Classifier.fit(trainingSet, trainingLabel)
    #return weights, bayes, SVM_Classifier
    return  SVM_Classifier


# 计算在测试集上的误差
def TestingSVM(testingSet, testingLabel, indexTupleList):
    testingSet = np.mat(testingSet)
    testingLabel = np.mat(testingLabel).transpose()
    numExp = len(testingLabel)
    # 去除参数列表中的方法名
    indexList = []
    for i in range(len(indexTupleList) - 1):
        indexList.append(indexTupleList[i])
    error = 0
    # (100,1)
    FinalPredict = np.mat(np.zeros((np.shape(testingLabel))))
    for indtup in indexList:
        clf = indtup
        # (100,1)
        svmPredict = np.mat(clf.predict(testingSet)).T
        # print(svmPredict)
        #selfPredict = testingSet * np.mat(weight) + bayes
        # print(np.shape(selfPredict))
        FinalPredict += (svmPredict )*1/len(indexList)
        
    FinalPredict = np.sign(FinalPredict)
    #print(FinalPredict[FinalPredict>0])
    for i in range(numExp):
        if(testingLabel[i] != FinalPredict[i]):
            error += 1
    errorRate = error/numExp
    return errorRate

# **********************************************************************************************************************
#              #########################         AdaBoost       ###########################
# **********************************************************************************************************************

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 此函数用于导入数据,返回列表类型的dataSet和labelSet
def loadAdaDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# 创建一个弱分类器，该分类器为只有一个节点的决策树，返回该分类器对于样本的分类结果
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 创建一个数组对应于所有的样本
    retArray=np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 不满足阈值的置为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 遍历分类器的所有可能输入，找到一个最佳决策树桩，返回包含分类树桩信息的字典，最小误差，最佳分类结果。
def bulidStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 正无穷大
    minError = np.inf
    # 对于每个feature
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        # j=[-1,0,,,,10]，对于每个feature的值
        for j in range(-1, int(numSteps)+1):
            # lt:less than, gt:greater than，对于阈值分类方法，到底是大于分类为反例还是小于分类为反例
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                # 经弱分类器分类之后的结果
                predictedVals = stumpClassify(
                    dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                # 与真实值相同的置0
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                # print("split:dim %d,thresh%.2f,thresh ineqal: %s,the weighted error is %.3f" %
                     # (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# 得到numIt个训练好的分类器，返回包含弱分类器信息的字典所构成的列表
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 每一次循环的借用上次的D向量
    for i in range(numIt):
        # 得到最佳决策树，错误率和估计向量
        bestStump, error, classEst = bulidStump(dataArr, classLabels, D)
        # print("D:", D.T)
        alpha = float(0.5 * np.log((1.0-error)/max(error, np.e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append((bestStump))
        # print("classEst:", classEst.T)
        # 使得预测的标签和真实标签相同的样本权重减小，相反的，不同的增大
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        # 各迭代循环得到的分类器对各样本分类的加权平均
        aggClassEst += alpha*classEst
        # print("aggClassEst:", aggClassEst.T)# 预测值
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))# true值为1
        # print("aggError:", aggErrors)
        errorRate = aggErrors.sum()/m
        # print("total error:", errorRate)
        if errorRate == 0.0:
            break
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20,
                                                    ),
                             algorithm="SAMME",
                             n_estimators=500, learning_rate=0.8)
    bdt.fit(dataArr, classLabels)
    # 列表里全是弱分类器
    return  bdt

# 应用adaBoostTrainDS训练出来的一组分类器分类数据
def adaClassify(testinfSet, classifierArr):#classifierArr是一组弱分类器组成的列表
    dataMatrix = np.mat(testinfSet)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range((len(classifierArr))):
        classEst=stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                               classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return aggClassEst

# 返回调包模型预测的结果
def stefanAdaClassify(testingSet, clf):
    midPredict = clf.predict(testingSet)
    midPredict = np.mat(midPredict).transpose()
    return midPredict

# 测试分类器
def TestingAdaBst(testingSet, testingLabel, indexLs):
    testingSet = np.mat(testingSet)
    testingLabel = np.mat(testingLabel).transpose()
    numExp = np.shape(testingLabel)[0]
    # 去除参数列表中的方法名
    indexList = []
    for i in range(len(indexLs)-1):
        indexList.append(indexLs[i])
    # print(indexList)
    error = 0
    FinallPredict = np.mat(np.zeros((numExp, 1)))
    for indtup in indexList:
        clf = indtup
        #predict = adaClassify(testingSet, weakClassArr)
        midPredict = stefanAdaClassify(testingSet, clf)
        FinallPredict += (midPredict)/ len(indexList)
    FinallPredict = np.sign(FinallPredict)
    for i in range(numExp):
        if(testingLabel[i] != FinallPredict[i]):
            error += 1
    errorRate = error/numExp
    return errorRate

# **********************************************************************************************************************
#               #########################         K Nearest Neighbor       ###########################
# ######################################################################################################################

import operator
from sklearn.neighbors import KNeighborsClassifier

# 此函数用于导入数据,返回列表类型的dataSet和labelSet
def loadKNNDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# 计算待分类实例与样本中各实例之间的距离，并升序排列，对于前k个，哪种分类所占的比重大，则认为实例属于哪种分类
def classify0(inX, dataSet, labels, k):
    # 获取行数
    dataSetSize = dataSet.shape[0]
    # 创建一个数组，按照第二个参数重复，计算与各样本点的距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 直观来看，横向相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 将distance从小到大排序，返回排序后的各元素的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}  # 新建字典
    # 对于给定的k，计算距离从小到大k个中不同种类的票数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # operator.itemgetter(1)是个函数，调用classCount.iteritems()遍历字典，按第二个阈值排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    # sorted()将字典类型分解为元组列表。
    return sortedClassCount[0][0]

# 缩放处理
def autoNorm(dataSet):
    # 归一化处理
    # dataSet = np.array(dataSet)
    # meanSet = np.mean(dataSet, axis=0)
    # midNorm = dataSet - meanSet
    # stdSet = np.std(midNorm, axis=0)
    # normalizationSet = midNorm / stdSet
    # 从列中选取最小值，返回每一列对应的最大值和最小值
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    # 按照数据集创建一个全为0的矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    # 获取行数，list.shape返回一个元组（m，n）
    m = dataSet.shape[0]
    # tile函数可以将（1,3）矩阵经过运算之后变成（1*m，3*1）矩阵
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 数组相除定义为各元素相除
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet

# 调包使用KNN
def trainKNN(dataSet, Labels, n_neighbors = 3):
    np.random.seed(0)
    knnClassifier = KNeighborsClassifier(n_neighbors)
    knnClassifier.fit(dataSet, Labels)
    return knnClassifier

# 测试KNN算法
def TestingKNN(testSet, testLabels, indexTuplelist):
    testSet = np.mat(testSet)
    testLabels = np.mat(testLabels).transpose()
    numExp = len(testLabels)
    # 去除参数列表中的方法名
    indexList = []
    for i in range(len(indexTuplelist) - 1):
        indexList.append(indexTuplelist[i])
    error = 0
    FinalPredict = np.mat(np.zeros((np.shape(testingLabel)))).transpose()
    for indtup in indexList:
        knnClassifier  = indtup
        midPredict = knnClassifier.predict(testSet)
        midPredict = np.mat(midPredict).transpose()
        FinalPredict += midPredict / len(indexTuplelist)
    FinalPredict = np.sign(FinalPredict)
    for i in range(numExp):
        if(FinalPredict[i] != testLabels[i]):
            error += 1
    error = error/numExp
    return error

# **********************************************************************************************************************
#                 #########################         Decision Tree       ###########################
# ######################################################################################################################

from sklearn import tree

# 此函数用于导入数据,返回列表类型的dataSet和labelSet
def loadTreeDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# 计算给定数据集的 Shannon Entropy
def calcShannonEnt(dataSet, Labels):
    # 计算数据集中实例的总数
    numEntries = np.shape(np.mat(dataSet))[0]
    labelCounts = {}  # 创建字典
    for i in range(numEntries):
        currentLabel = Labels[i]
        # 如果以currentLabels为键的键值对在在字典中不存在
        if currentLabel not in labelCounts.keys():
            # 新建一个键值对{"currentLabel":0}
            labelCounts[currentLabel] = 0
        # 统计对于每个标签，数据集中共有多少个，
        labelCounts[currentLabel] += 1
        # 最终在字典中以{"标签":个数}形式存放
    shannonEnt = 0.0
    for key in labelCounts:  # 遍历字典，键值放在key中
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt

# axis为分类特征，当该特征等于value时，将满足特征的样本分离出来，并将该特征删除
def splitDataSet(dataSet, Labels, axis, value):
    retDataSet = []
    retLabel = []
    for i in range(len(Labels)):
        if dataSet[i][axis] == value:
            # 相当于对满足分类特征的实例，将分类特征消去
            reducedFeatVec = dataSet[i][:axis]
            # 将括号内的列表元素加到reducedFeatVec中
            reducedFeatVec.extend(dataSet[i][axis + 1:])
            # 将整个列表加入retDataSet中，形成二维列表
            retDataSet.append(reducedFeatVec)
            retLabel.append(Labels[i])
    return retDataSet, retLabel

# 计算最好的分类特征
def chooseBestFeatureToSplit(dataSet, Labels):
    numFeatures = len(dataSet[0])
    baseEntropy = calcShannonEnt(dataSet, Labels)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建集合，过滤掉重复的标签
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet ,subLabels = splitDataSet(dataSet, Labels, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet, subLabels)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 返回分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树,以字典形式保存，featLabel为每个feature对应的含义，这里为其对应的标号。
def createTree(dataSet, Labels, featLabel):
    classList = Labels
    if classList.count(classList[0]) == len(classList):
        # count函数，数classList中有多少个跟第一个一样，如果全部一样，分类结束
        return classList[0]
    # 因为遍历完一个特征值，即将该特征值消去，最后长度为一时即是label
    if len(dataSet[0]) == 1:
        # 调用函数，返回占比最大的label值
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, Labels)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    bestFeatLabel = featLabel[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (featLabel[bestFeat])
    for value in uniqueVals:
        subFeatLabels = featLabel[:]
        subDataset, subLabels = splitDataSet(dataSet, Labels, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataset, subLabels, subFeatLabels)
    # 调包使用决策树
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(dataSet, Labels)
    return myTree, clf

# 应用树分类
def TreeClassifier(inputTree, featLabel, testSet):
    firstStr = (list(inputTree.keys()))[0]
    secondDict = inputTree[firstStr]
    # 使用index方法查找当前列表中第一个匹配firstStr变量的元素
    featIndex = featLabel.index(firstStr)
    testSet = np.array(testSet)
    for testVec in testSet:
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = TreeClassifier(secondDict[key], featLabel, testVec)
                else:
                    classLabel = secondDict[key]
                predictLabel.append(classLabel)
    return predictLabel

# 测试决策树算法
def TestingTree(testingSet, testingLabel, indexTupleList):
    testingSet = np.mat(testingSet)
    featLabel = []
    for i in range(testingSet.shape[1]):
        featLabel.append(str(i))
    testingLabel = np.mat(testingLabel).transpose()
    numExp = len(testingLabel)
    # 去除参数列表中的方法名
    indexList = []
    for i in range(len(indexTupleList) - 1):
        indexList.append(indexTupleList[i])
    error = 0
    FinalPredict = np.mat(np.zeros((np.shape(testingLabel))))
    for indtup in indexList:
        myTree, clf = indtup
        treePredict = np.mat(clf.predict(testingSet)).T
        selfPredict = np.mat(TreeClassifier(myTree, featLabel, testingSet)).T
        # print(np.shape(selfPredict))
        FinalPredict += (treePredict * 0.5 + selfPredict * 0.5)/len(indexList)
    # print(FinalPredict)
    for i in range(numExp):
        if(FinalPredict[i] * testingLabel[i] <= 0):
            error += 1
    errorRate = error/numExp
    return errorRate

# **********************************************************************************************************************
#               #########################         Logistic Regression       ###########################
# ######################################################################################################################

from sklearn.linear_model import LogisticRegression

# 此函数用于导入数据,返回列表类型的dataSet和labelSet
def loadLgreMDataSet(fileName):
    dataSet = []
    labelSet = []
    fileName = path + fileName
    # 得到每行中特征数
    numFeat = len(open(fileName).readline().strip().split())
    fr = open(fileName)
    for line in fr.readlines():
        currLine = []
        lineArr = line.strip().split()
        for i in range(numFeat - 1):
            currLine.append(float(lineArr[i]))
        dataSet.append(currLine)
        if(float(lineArr[-1]) == 0):
            labelSet.append(float(-1))
        else:
            labelSet.append(float(lineArr[-1]))
    # print(np.array(dataSet).shape)
    # dataSetN = normalLization(dataSet)
    return dataSet, labelSet

# Sigmoid 函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# 梯度上升算法
def gradDescent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.005
    maxCycles = 500
    weights1 = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights1)
        error = (labelMat - h)
        weights = weights1 + alpha * dataMatrix.transpose() * error
    return weights

# 随机梯度上升算法，训练速度快，但不稳定
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    dataMatrix = np.array(dataMatrix)  # 二维列表转数组
    classLabels = np.array(classLabels)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

# 梯度下降算法，速度慢，精度与梯度上升算法相近
def gradDescent1(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 生成matrix
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.005
    # 梯度下降算法
    c = np.ones((m, 1))
    weights = np.ones((n, 1))
    L1 = float(sum(np.square(sigmoid(dataMatrix * weights) - labelMat)))
    i = 0
    print(L1)
    while (True):
        i += 1
        L2 = L1
        h = sigmoid(dataMatrix * weights)
        d = np.multiply(labelMat - h, c - h)
        e = np.multiply(d, h)
        weights = weights + alpha * dataMatrix.T * e
        L1 = float(sum(np.square(sigmoid(dataMatrix * weights) - labelMat)))
        if (L2 - L1 < 0.000001):
            break
        if (i > 1000000):
            break
    return weights

# 调包使用LogisticRegression
def trainLgre(trainingSet, trainingLabels):
    lr = LogisticRegression(penalty='l2', random_state=0, C=1.0,
                            solver='lbfgs', max_iter=1000, multi_class='multinomial', class_weight='balanced')
    lr.fit(trainingSet, trainingLabels)
    return lr

# 测试Logistic Regression
def TestingLgre(testSet, testLabels, indexTuplelist):
    testSet = np.mat(testSet)
    testLabels = np.mat(testLabels).transpose()
    numExp = len(testLabels)
    # 去除参数列表中的方法名
    indexList = []
    for i in range(len(indexTuplelist) - 1):
        indexList.append(indexTuplelist[i])
    error = 0
    FinalPredict = np.mat(np.zeros((np.shape(testingLabel)))).transpose()
    for indtup in indexList:
        lgreClassifier = indtup
        midPredict = lgreClassifier.predict(testSet)
        midPredict = np.mat(midPredict).transpose()
        FinalPredict += midPredict / len(indexList)
    FinalPredict = np.sign(FinalPredict)
    for i in range(numExp):
        if (FinalPredict[i] != testLabels[i]):
            error += 1
    error = error / numExp
    return error


if __name__ =="__main__":
    trainingDataFiles = ["MADL434.txt","MADL544.txt"]
    testingDataFiles = ["MADL434.txt","MADL544.txt","MADL322.txt","MADL332.txt","MADL433.txt","MADL443.txt"]

    # Support Vector Machine
    if(ModelSelction == 1):
        # 存放由训练数据训练出的分类器模型，是元组元素构成的列表
        index = []
        print("Model for breakout detection is Support Vector Machine")
        for i in range(len(trainingDataFiles)):
            trainingData, trainingLabel = loadSVMDataSet(trainingDataFiles[i])
            indextuple = StefanSVM(trainingData, trainingLabel, (0.6, 0.001, 40))
            index.append(indextuple)
        # 加入方法标志
        index.append("Support_Vector_Machine")
        # print(index)
        # 将模型保存到本地
        fr = open("/home/stefan/PythonFiles/Models/SVMModel_{}.model".format(currTime), mode='wb')
        pickle.dump(index, fr)
        print("The Model You trained has been saved as SVMModel_{}.model".format(currTime))
        fr.close()
        # 读取文件
        # fr1 = open("Model.txt", mode='rb')
        # index1 = pickle.load(fr1)
        for testingSetName in testingDataFiles:
            testingData, testingLabel = loadTestDataSet(testingSetName)
            errorRate = TestingSVM(testingData, testingLabel, index)
            print("Error rate in TestingSet {} is {}".format(testingSetName, errorRate))

    # Adaptive Boosting
    if(ModelSelction == 2):
        # 存放由训练数据训练出的分类器模型，是元组元素构成的列表
        index = []
        print("Model for breakout detection is Adaptive Boosting")
        for i in range(len(trainingDataFiles)):
            trainingData, trainingLabel = loadAdaDataSet(trainingDataFiles[i])
            indextuple = adaBoostTrainDS(trainingData, trainingLabel)
            index.append(indextuple)
        # 加入方法标志
        index.append("Adaptive_Boosting")
        # 将模型保存到本地
        fr = open("/home/stefan/PythonFiles/Models/AdaBoostModel_{}.model".format(currTime), mode='wb')
        pickle.dump(index, fr)
        print("The Model You trainned has been saved as AdaBoostModel_{}.model".format(currTime))
        fr.close()
        # 读取文件
        # fr1 = open("AdaBoostModel_04_25_13_11.model", mode='rb')
        # index1 = pickle.load(fr1)
        for testingSetName in testingDataFiles:
            testingData, testingLabel = loadTestDataSet(testingSetName)
            errorRate = TestingAdaBst(testingData, testingLabel, index)
            print("Error rate in TestingSet {} is {}".format(testingSetName, errorRate))

    # K Nearest Neighbor
    if(ModelSelction == 3):
        index = []  # 存放由训练数据训练出的分类器模型，是元组元素构成的列表
        print("Model for breakout detection is K Nearest Neighbor ")
        for i in range(len(trainingDataFiles)):
            trainingData, trainingLabel = loadKNNDataSet(trainingDataFiles[i])
            indextuple = trainKNN(trainingData, trainingLabel)
            index.append(indextuple)
        # 加入方法标志
        index.append("K_Nearest_Neighbor")
        # 将模型保存到本地
        fr = open("/home/stefan/PythonFiles/Models/KNNModel_{}.model".format(currTime), mode='wb')
        pickle.dump(index, fr)
        print("The Model You trainned has been saved as KNNModel_{}.model".format(currTime))
        fr.close()
        # 读取文件
        # fr1 = open("AdaBoostModel.model", mode='rb')
        # index1 = pickle.load(fr1)
        for testingSetName in testingDataFiles:
            testingData, testingLabel = loadTestDataSet(testingSetName)
            errorRate = TestingKNN(testingData, testingLabel, index)
            print("Error rate in TestingSet {} is {}".format(testingSetName, errorRate))

    # Decision Tree
    if(ModelSelction == 4):
        index = []  # 存放由训练数据训练出的分类器模型，是元组元素构成的列表
        print("Model for breakout detection is Decision Tree ")
        for i in range(len(trainingDataFiles)):
            trainingData, trainingLabel = loadTreeDataSet(trainingDataFiles[i])
            featLabel = []
            for j in range(len(trainingData[0])):
                featLabel.append(str(i))
            indextuple = createTree(trainingData, trainingLabel, featLabel)
            index.append(indextuple)
        # 加入方法标志
        index.append("Decision_Tree")
        # 将模型保存到本地
        fr = open("/home/stefan/PythonFiles/Models/DecisionTreeModel_{}.model".format(currTime), mode='wb')
        pickle.dump(index, fr)
        print("The model you trainned has been saved as DecisionTreeModel_{}.model".format(currTime))
        fr.close()
        # 读取文件
        # fr1 = open("AdaBoostModel.model", mode='rb')
        # index1 = pickle.load(fr1)
        for testingSetName in testingDataFiles:
            predictLabel = []
            testingData, testingLabel = loadTestDataSet(testingSetName)
            errorRate = TestingTree(testingData, testingLabel, index)
            print("Error rate in TestingSet {} is {}".format(testingSetName, errorRate))

    # Logistic Regression
    if(ModelSelction == 5):
        # 存放由训练数据训练出的分类器模型，是元组元素构成的列表
        index = []
        print("Model for breakout detection is Logistic Regression")
        for i in range(len(trainingDataFiles)):
            trainingData, trainingLabel = loadLgreMDataSet(trainingDataFiles[i])
            indextuple = trainLgre(trainingData, trainingLabel)
            index.append(indextuple)
        # 加入方法标志
        index.append("Logistic_Regression")
        # print(index)
        #将模型保存到本地
        fr = open("/home/stefan/PythonFiles/Models/Logistic_RegressionModel_{}.model".format(currTime), mode='wb')
        pickle.dump(index, fr)
        print("The Model You trained has been saved as Logistic_RegressionModel_{}.model".format(currTime))
        fr.close()
        # 读取文件
        # fr1 = open("Model.txt", mode='rb')
        # index1 = pickle.load(fr1)
        for testingSetName in testingDataFiles:
            testingData, testingLabel = loadTestDataSet(testingSetName)
            errorRate = TestingLgre(testingData, testingLabel, index)
            print("Error rate in TestingSet {} is {}".format(testingSetName, errorRate))




