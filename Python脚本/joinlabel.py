import numpy as np
import sys

sys.path.append("C:/Users/yxldenanren/Desktop/")

fr1 = open("MAD322label.txt")
Label = []
for label in fr1.readlines():
    label = label.strip()
    if(float(label)==0):
        Label.append(float(-1))
    else:
        Label.append(float(1))

fr = open("MAD322.txt")

dataSet = []
for line in fr.readlines():
       currLine = []
       line = line.strip().split('\t')
       for i in range(2):
              currLine.append(float(line[i]))
       dataSet.append(currLine)
       
f = open("MADL322.txt", 'w')
for j in range(len(Label)):
    for i in range(2):
        f.write(str(dataSet[j][i]))
        f.write('\t')
    f.write(str(Label[j]))
    f.write('\n')
f.close()
