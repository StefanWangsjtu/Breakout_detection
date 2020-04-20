## *********************************************************************************************************************
## This Python File is Built for Load Model You Have Trained and Apply it to Breakout Detection in Fast EDM Drilling
## *********************************************************************************************************************


#from PythonQt import *
import sys
sys.path.append("/usr/local/lib/python3.5/dist-packages")
#import pickle
#import numpy as np
#from sklearn import svm
#from os import listdir
#import sys
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#import operator
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import tree
#from sklearn.linear_model import LogisticRegression

from PythonQt import *
import numpy as np

def hello():
    a = np.ones((3,4))
    print(a)

## 获取文件名
#modelPath = "/home/stefan/PythonFiles/Models/"
#modelList = listdir(modelPath)[-1]
#print(modelList)
#modelList = modelPath + modelList
## 读文件
#fr = open(modelList, "rb")
#indexTupleList = pickle.load(fr)
#indexList = []
#for i in range(len(indexTupleList)-1):
#    indexList.append(indexTupleList[i])

## 获取模型名称
#def getModelName():
#    return indexTupleList[-1]
#testingSet = []

## 调用分类器
#def callClassifier(modelName):
#    print("enter callClassifier")
#    print(testingSet)
#    if(modelName == "Support_Vector_Machine"):
#        return Support_Vector_Machine(testingSet)

#    if(modelName == "Adaptive_Boosting"):
#        return Adaptive_Boost(testingSet)

#    if(modelName == "K_Nearest_Neighbor"):
#        a = K_Nearest_Neighbor(testingSet)
#        return a

#    if(modelName == "Logistic_Regression"):
#        a = Logistic_Regression(testingSet)
#        return a

## 合并数据
#def creatData(a):
#    testingSet.append(a)

## 清除列表
#def clearData():
#    testingSet.clear()

#fr1 = open("/home/stefan/PythonFiles/label.txt",'w')
#def creatLabel(label):
#    fr1.write(str(label))
#    fr1.write(str('\n'))
#def closefile():
#    fr1.close()


## SVM分类器
#def Support_Vector_Machine(testSet):
#    print("Support_Vector_Machine")
#    final = 0
#    m = len(testSet)
#    testSet = np.mat(testSet).reshape((1, m))
#    for indtup in indexList:
#        clf = indtup
#        svmPredict = clf.predict(testSet)[0]
#        print(svmPredict)
#        #selfPredict = testingSet * np.mat(weight) + bayes
#        final += (svmPredict)*1/len(indexList)

#    return final


## 创建一个弱分类器，该分类器为只有一个节点的决策树，返回该分类器对于样本的分类结果
#def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
#    # 创建一个数组对应于所有的样本
#    retArray=np.ones((np.shape(dataMatrix)[0], 1))
#    if threshIneq == 'lt':
#        # 不满足阈值的置为-1
#        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
#    else:
#        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
#    return retArray

## 应用adaBoostTrainDS训练出来的一组分类器分类数据,classifierArr是一组弱分类器组成的列表
#def adaClassify(testSet, classifierArr):
#    dataMatrix = np.mat(testSet)
#    m = np.shape(dataMatrix)[0]
#    aggClassEst = np.mat(np.zeros((m, 1)))
#    for i in range((len(classifierArr))):
#        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
#                               classifierArr[i]['ineq'])
#        aggClassEst += classifierArr[i]['alpha'] * classEst
#    return aggClassEst


## AdaBoost分类器
#def Adaptive_Boost(testSet):
#    print("enter Adaptive_Boost")
#    m = len(testSet)
#    testSet = np.mat(testingSet).reshape((1, m))
#    final = 0
#    for indtup in indexList:
#        clf = indtup
#        #predict = adaClassify(testSet, weakClassArr)[0][0]
#        midPredict = clf.predict(testSet)[0]
#        final += (midPredict)/len(indexList)
#    final = np.sign(final)
#    return final


## 测试KNN算法
#def K_Nearest_Neighbor(testSet):
#    print("K_Nearest_Neighbor")
#    m = len(testSet)
#    testSet = np.mat(testSet).reshape((1, m))
#    final = 0
#    for indtup in indexList:
#        knnClassifier  = indtup
#        midPredict = knnClassifier.predict(testSet)[0]
#        final += midPredict / len(indexTuplelist)
#    final = np.sign(final)
#    return final


## 测试Logistic Regression
#def Logistic_Regression(testSet):
#    print("enter Logistic Regression")
#    m = len(testSet)
#    final = 0
#    testSet = np.mat(testSet).reshape((1, m))
#    for indtup in indexList:
#        lgreClassifier = indtup
#        midPredict = lgreClassifier.predict(testSet)[0]
#        final += midPredict/len(indexList)
#    return final










