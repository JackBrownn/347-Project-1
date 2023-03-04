import numpy as np  
import math
import pandas as pd
import matplotlib.pyplot as plt


###################### Mean #######################
def mean(arr): # Function to calculate multivariate mean of dataset
    num_rows, num_cols = arr.shape # stores number of rows and columns
    ret = []
    for i in range(num_cols):
        tot = 0
        for j in range(num_rows):
            tot = tot + arr[j,i]
        ret.append(tot/num_rows) # appends column mean to dataset
    return(np.array(ret))


###################### Covariance #######################
def covariance(v1,v2): # Function to find covariance of two vectors
    m1 = np.mean(v1) 
    m2 = np.mean(v2)
    c1 = np.shape(v1)[0]
    ret = 0
    for i in range(c1):
        ret = ret +((v1[i]-m1)*(v2[i]-m2))
    return (ret/(c1-1))

###################### Sample/Total Variance ##############
def sampleVariance(arr): # Function to find sample variance of a vector
    arrayMean = np.mean(arr) # Stores mean of array
    total = 0
    for i in range(arr.size):
        total += ((arr[i] - arrayMean) ** 2)
        
    return total / (arr.size - 1)

def totalVariance(arr): # Function to find the total variance of a matrix
    cols = np.shape(arr)[1]
    totalVar = 0
    
    for column in range(cols):
        totalVar += sampleVariance(arr[:,column]) # Adds sample variance to total variance
        
    return totalVar

###################### Negative Covariance #######################
def negCovariance(arr): # Function to find the number of negative covariances
    cols = np.shape(arr)[1] # Stores number of columns in dataset
    num = 0

    for row in range(cols):
        for column in range(row, cols):
            if (covariance(arr[:,row], arr[:,column]) < 0):
                num += 1

    return num

###################### Correlation  #######################
def correlation(v1,v2): # Function to find the correlation between two vectors
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
def correlationGreater(arr, g): # Function to determine how many correlations are greater than a specific value
    cols = np.shape(arr)[1]
    num = 0

    for row in range(cols):
        for column in range(row, cols):
            if (correlation(arr[:,row], arr[:,column]) >= g):
                num += 1

    return num


###################### Range Normalization #######################
def rangeNormalization(arr, minimum, maximum): # Function using range normalization to normalize data
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
def standardNormalization(arr): # Function using z-score normalization to normalize data
    rows = arr.shape[0] # stores number of rows in data
    cols = arr.shape[1] # stores number of columns in data

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
def covarianceMatrix(arr): # Function to find the covariance matrix of a datset
    cols = arr.shape[1]
    covMat = np.empty([cols, cols]) # creates empty 2D array for covariance matrix

    for row in range(cols):
        for column in range(cols):
            covMat[row, column] = covariance(arr[:,column], arr[:,row]) # stores covariance to covariance matrix

    return covMat

###################### Label Encoding #######################
def labelEncoding(catArr): # Function using Label Encoding on categorical variables
  
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
def oneHotEncoding(catArr): # Function using One-Hot Encoding on categorical variable

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
numericalAttributes = data.iloc[:, [2, 4, 10, 11, 12]]
numericalAttributes = np.array(numericalAttributes)
data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]] = labelEncoding(categoricalAttributes)
data = np.array(data)


print('Multivariate Mean: ', mean(data))
print('Covariance Matrix: ', covarianceMatrix(data))
print('Scatter Plots: ')
print('Multivariate Mean: ')
=======
#data = 
>>>>>>> 589bb1a62c6bfd36439d99f61ffc37628339f84c

############################ Z-Score Normalization ############################

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

# TOTAL VARIANCE
print(totalVariance(data))

# TOTAL VARIANCE OF 5 GREATEST
print(str(12089776858.609715 + 4105580.4150943398 + 173946.83018867925 + 26.41509433962264 + 0.7169811320754716))
