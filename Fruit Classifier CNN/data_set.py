import cv2
import glob
import numpy as np
import h5py
import os

test_set_x = []
test_set_y = []

train_set_x = []
train_set_y = []

################ Creating Training Data set ###############################

files = glob.glob ("C:\Users\User\Desktop\Data\Apple Green\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[0] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

print(train_set_y[0], train_set_y[10], train_set_y[2])

files = glob.glob ("C:\Users\User\Desktop\Data\Apple Red\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[1] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Apple Red Yellow\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[2] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Banana\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[3] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape


files = glob.glob ("C:\Users\User\Desktop\Data\Banana Red\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[4] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Grape Blue\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[5] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Grape Pink\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[6] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Grape White\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[7] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Guava\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[8] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Mango\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[9] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Orange\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[10] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)
print np.array(train_set_x).shape, np.array(train_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\Papaya\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[11] = 1
    
    train_set_x.append (image)
    train_set_y.append (lable)

print np.array(train_set_x).shape, np.array(train_set_y).shape

##########################  Creating Test data set  ##########################

files = glob.glob ("C:\Users\User\Desktop\Data\TApple Green\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[0] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape


files = glob.glob ("C:\Users\User\Desktop\Data\TApple Red\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[1] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape


files = glob.glob ("C:\Users\User\Desktop\Data\TApple Red Yellow\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[2] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TBanana\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[3] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TGrape Blue\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[5] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TGrape Pink\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[6] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TGrape White\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[7] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape


files = glob.glob ("C:\Users\User\Desktop\Data\TPapaya\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[11] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TGuava\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[8] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TMango\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[9] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape

files = glob.glob ("C:\Users\User\Desktop\Data\TOrange\*.jpg")
for myFile in files:
    
    image = cv2.imread (myFile)
    lable = np.full((12), 0)
    lable[10] = 1
    
    test_set_x.append (image)
    test_set_y.append (lable)
print np.array(test_set_x).shape, np.array(test_set_y).shape


hf = h5py.File('data.h5', 'w')
hf.create_dataset('train_set_x', data=train_set_x)
hf.create_dataset('train_set_y', data=train_set_y)
hf.create_dataset('test_set_x', data=test_set_x)
hf.create_dataset('test_set_y', data=test_set_y)
hf.create_dataset('classes', data= 12)
