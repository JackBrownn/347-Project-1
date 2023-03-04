import numpy as np  
import math
import pandas as pd
import matplotlib.pyplot as plt


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

###################### Negative Covariance #######################
def negCovariance(arr):
    cols = np.shape(arr)[1]
    num = 0

    for row in range(cols):
        for column in range(row, cols):
            if (covariance(arr[:,row], arr[:,column]) < 0):
                num += 1

    return num

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

###################### Correlation Greater #######################
def correlationGreater(arr, g):
    cols = np.shape(arr)[1]
    num = 0

    for row in range(cols):
        for column in range(row, cols):
            if (correlation(arr[:,row], arr[:,column]) >= g):
                num += 1

    return num


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
<<<<<<< HEAD
def covarianceMatrix(arr):
    cols = arr.shape[1]
    covMat = np.empty([cols, cols])

    for row in range(cols):
        for column in range(cols):
            covMat[row, column] = covariance(arr[:,column], arr[:,row])

    return covMat
=======
#def covariance(arrA, arrB):
#    colsA = arrA.size
#    totalX = 0
#    totalY = 0
#    for index in range(0, colsA):
#        totalX += arrA[index]
#        totalY += arrB[index]
#    meanX = totalX/colsA
#    meanY = totalY/colsA
#    topVal = float(0)
#    for index in range(0, colsA):
#        topVal += ((arrA[index]-meanX)*(arrB[index]-meanY))
#    covarianceAB = topVal / colsA

#    return covarianceAB
    
>>>>>>> 589bb1a62c6bfd36439d99f61ffc37628339f84c

#def covarianceMatrix(arr):
#    rows = arr.shape[0]
#    cols = arr.shape[1]

    #if rowindex == columnindex set to variance (covariance(arrX, arrX))
    #in covariance send (covariance(arr[row] and arr[column])
#    retVal = np.eye(rows, cols)
#    for indexRow in range(0, rows):
#        for indexColumn in range(0, cols):
#            retVal[indexRow][indexColumn] = covariance(arr[indexRow], arr[indexColumn])
#    return retVal

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

<<<<<<< HEAD
#import os
#print(os.getcwd())  
from numpy import genfromtxt

data =  pd.read_csv('adultTest.data', sep=",") 

categoricalAttributes = data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]]
data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]] = labelEncoding(categoricalAttributes)
data = np.array(data)


print('Multivariate Mean: ', mean(data))
print('Covariance Matrix: ', covarianceMatrix(data))
print('Scatter Plots: ')
print('Multivariate Mean: ')
=======
#data = 
>>>>>>> 589bb1a62c6bfd36439d99f61ffc37628339f84c

################################ 5 Scatterplots ################################

plt.scatter(np.array(data[:,1]), np.array(data[:,3]))
plt.xlabel("Work-Class")
plt.ylabel("Education Level")
plt.show()

plt.scatter(np.array(data[:,3]), np.array(data[:,7]))
plt.xlabel("Education Level")
plt.ylabel("Relationship Status")
plt.show()

plt.scatter(np.array(data[:,0]), np.array(data[:,12]))
plt.xlabel("Age (Years)")
plt.ylabel("Hours Worked Per Week")
plt.show()

plt.scatter(np.array(data[:,6]), np.array(data[:,9]))
plt.xlabel("Occupation")
plt.ylabel("Sex")
plt.show()

plt.scatter(np.array(data[:,13]), np.array(data[:,5]))
plt.xlabel("Native Country")
plt.ylabel("Marital Status")
plt.show()


# CORRELATION GREATER THAN .5
print(correlationGreater(data, .5))

# NEGATIVE SAMPLE COVARIANCE
print(negCovariance(data))
