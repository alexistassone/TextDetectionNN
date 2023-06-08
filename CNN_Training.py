import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get list of directories in our path
path = 'myData'
testRatio = 0.2
validationRatio = 0.2
images = []
classNum = []
myList = os.listdir(path)
print("Total Num of Classes Detected:", len(myList))
numOfClasses = len(myList)
print("Importing Classes...")

# Read each image corresponding to each number and put it in a lista
for x in range(0, numOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        currImg = cv2.imread(path+"/"+str(x)+"/"+y)
        # Resize image for network
        currImg = cv2.resize(currImg, (32, 32))
        images.append(currImg)
        classNum.append(x)
    print(x, end = " ")
print(" ")
print("Total Images in Images List:", len(images))
print("Total IDs in classNum List:", len(classNum))

images = np.array(images)
classNum = np.array(classNum)
print(images.shape)
# print(classNum.shape)

# Split the data into training, testing, and validating set
X_train, X_test, y_train, y_test = train_test_split(images, classNum, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, numOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0, numOfClasses), numOfSamples)
plt.title("Num of Images for Each Class")
plt.xlabel("Class ID")
plt.ylabel("Num of Images")
plt.show()
