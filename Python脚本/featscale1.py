import numpy as np
import sys

sys.path.append("C:/Users/yxldenanren/Desktop/")

fr = open("E:\\毕业设计\\实验数据\\截断数据\\544.txt")

dataSet = []
for line in fr.readlines():
       currLine = []
       line = line.strip().split(' ')
       for i in range(2):
              currLine.append(float(line[i]))
       dataSet.append(currLine)

dataSet = np.array(dataSet)
dataSet[:, 0] = (2500-dataSet[:, 0])*(7/12)/1000
dataSet[:, 1] = dataSet[:, 1]/160


f = open("E:\\毕业设计\\实验数据\\截断数据\\MAD4431.txt", 'w')
'''
for j in range(len(dataSet)-4):
       for i in range(j,j+5):
              for m in range(2):
                     f.write(str(dataSet[i][m]))
                     f.write('\t')      
       f.write('\n')
'''

for j in range(len(dataSet)-4):
       a = np.mean(dataSet[j:j+5],axis=0)
       for m in range(2):
              f.write(str(a[m]))
              f.write('\t')
       f.write('\n')

f.close()
