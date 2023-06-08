import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# Settings
path = 'myData'
pathLabels = 'labels.csv'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)

# Get list of directories in our path
count = 0
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
        currImg = cv2.resize(currImg, (imageDimensions[0], imageDimensions[1]))
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

plt.figure(figsize=(10, 5))
plt.bar(range(0, numOfClasses), numOfSamples)
plt.title("Num of Images for Each Class")
plt.xlabel("Class ID")
plt.ylabel("Num of Images")
# plt.show()

# Pre-process images
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# Add depth to images
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Generate images to be augmented
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, numOfClasses)
y_test = to_categorical(y_test, numOfClasses)
y_validation = to_categorical(y_validation, numOfClasses)

#Create model
def myModel():
    numOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    numOfNode = 500

    model = Sequential()
    model.add((Conv2D(numOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(numOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(numOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(numOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(numOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# Run training
