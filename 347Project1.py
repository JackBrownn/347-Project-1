import numpy as np  
import math
import pandas as pd


###################### Mean #######################
def mean(arr):
    num_rows, num_cols = arr.shape
    ret = []
    for i in range(num_cols):
        tot = 0
        for j in range(num_rows):
            tot = tot + arr[j,i]
        ret.append(tot/num_rows)
    return(np.array(ret))


###################### Covariance #######################
def covariance(v1,v2):
    m1 = np.mean(v1)
    m2 = np.mean(v2)
    c1 = np.shape(v1)[0]
    ret = 0
    for i in range(c1):
        ret = ret +((v1[i]-m1)*(v2[i]-m2))
    return (ret/(c1-1))


###################### Correlation  #######################
def correlation(v1,v2):
    c1 = np.shape(v1)[0]
    m1 = 0
    m2 = 0
    for i in range(c1):
        m1 = m1 +v1[i]
        m2 = m2 +v2[i]
    m1=m1/c1
    m2=m2/c1
    
    for i in range(c1):
        v1[i] = v1[i]-m1
        v2[i] = v2[i]-m2
    
    top = 0
    for i in range(c1):
        top = top + v1[i]*v2[i]
    
    l1 = 0
    for i in range(c1):
        l1 = l1 + v1[i]*v1[i]
    l1 = math.sqrt(l1)
    
    l2 = 0
    for i in range(c1):
        l2 = l2 + v2[i]*v2[i]
    l2 = math.sqrt(l2)
    
    return(top/(l1*l2))


###################### Range Normalization #######################
def rangeNormalization(arr, minimum, maximum):
    difference = maximum - minimum
    rows = arr.shape[0]
    cols = arr.shape[1]

    retVal = np.eye(rows, cols)
    
    for indexR in range(0, rows):
        divVal = max(arr[indexR]) - min(arr[indexR])
        for entry in range(0, cols):
            newVal = ((arr[indexR, entry] - min(arr[indexR])) * difference)
            newVal = newVal/divVal + minimum
            retVal[indexR, entry] = newVal
    return(np.array(retVal))


###################### Standard Normalization #######################
def standardNormalization(arr):
    rows = arr.shape[0]
    cols = arr.shape[1]

    retVal = np.eye(rows, cols)
    
    for indexR in range(0, rows):
        divVal = max(arr[indexR]) - min(arr[indexR])
        for entry in range(0, cols):
            newVal = ((arr[indexR, entry] - min(arr[indexR])))
            newVal = newVal/divVal
            retVal[indexR, entry] = newVal
    return(np.array(retVal))


###################### Covariance Matrix #######################
def covarianceMatrix(arr):
    rows = arr.shape[0]
    cols = arr.shape[1]

    #if rowindex == columnindex set to variance (covariance(arrX, arrX))
    #in covariance send (covariance(arr[row] and arr[column])
    retVal = np.eye(rows, cols)
    for indexRow in range(0, rows):
        for indexColumn in range(0, cols):
            retVal[indexRow][indexColumn] = covariance(arr[indexRow], arr[indexColumn])
    return retVal


###################### Label Encoding #######################
def labelEncoding(catArr):
  
    catArr = np.array(catArr)
    indArr = np.split(catArr, len(catArr))  #splits the 2D into individual 1D arrays
    finalEncode = []


    for item in indArr:    #loops thourgh every individual array
        item = item.ravel()  #removes unnecessary brackets

        uniqueSortedArr = np.unique(item)

        newArr =[]

        for i in range(len(item)):
            value = np.where(uniqueSortedArr == item[i])[0][0]
            newArr.append(value)

        finalEncode.append(newArr)

    return(np.array(finalEncode))


###################### One Hot Encoding #######################
def oneHotEncoding(catArr):

    catArr = np.array(catArr)
    indArr = np.split(catArr, len(catArr))  #splits the 2D into individual 1D arrays
    finalEncode = []

    for item in indArr:    #loops thourgh every individual array
      item = item.ravel()  #removes unnecessary brackets

      rows = len(item)
      cols = len(np.unique(item))
      encodedArr = [[0]*rows for i in range(cols)]

      uniqueArr = [[0]*1 for i in range(cols)]
      for i in range(len(uniqueArr)):
        for j in range(len(uniqueArr[i])):
          uniqueArr[i][j] = np.unique(item)[i]


      for i in range(len(encodedArr)):
        for j in range(len(encodedArr[i])):
          if([item[j]] == uniqueArr[i]):
            encodedArr[i][j] = 1


      for arr in encodedArr:
        finalEncode.append(arr)
        
    
    return(finalEncode)



############################################## Part 3 Answers ##########################################

#import os
#print(os.getcwd())  
from numpy import genfromtxt

#read in data: must be in the 347-project-1 directory to work
data =  pd.read_csv('adultTest.data', sep=",") 

#label encode the categorical attributes and turn matrix in to numpy array
categoricalAttributes = data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]]
numericalAttributes = data.iloc[:, [2, 4, 10, 11, 12]]
numericalAttributes = np.array(numericalAttributes)


data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]] = labelEncoding(categoricalAttributes)
data = np.array(data)




#Print answers to Part 3
print('Multivariate Mean: ', mean(data))
print('Covariance Matrix: ', covarianceMatrix(data))
print('Scatter Plots: ', )
print('Multivariate Mean: ')
print("blah")

#print(numericalAttribute2)



zScore = standardNormalization(numericalAttributes)
zScore2 = []
zScore4 = []
zScore10 = []
zScore11 = []
zScore12 = []
for i in range(len(zScore)):
    zScore2.append(zScore[i][0])
    zScore4.append(zScore[i][1])
    zScore10.append(zScore[i][2])
    zScore11.append(zScore[i][3])
    zScore12.append(zScore[i][4])


print(covariance(zScore2, zScore4))
print(covariance(zScore2, zScore10))
print(covariance(zScore2, zScore11))
print(covariance(zScore2, zScore12))

print(covariance(zScore4, zScore10))
print(covariance(zScore4, zScore11))
print(covariance(zScore4, zScore12))

print(covariance(zScore10, zScore11))
print(covariance(zScore10, zScore12))

print(covariance(zScore11, zScore12))


