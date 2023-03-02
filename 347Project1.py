import numpy as np  
import math


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


    return(finalEncode)



############################################## Part 3 Answers ##########################################

#data = 