#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras, sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

# Making Command line arguments optional
# Tweeking Model 
ker_size = 2
batch_size_passed = 1024
no_of_epochs = 1
crp_count = 1
fc_count = 1
if len(sys.argv) == 2:
    ker_size = int(sys.argv[1])
elif len(sys.argv) == 3:
    ker_size = int(sys.argv[1])
    batch_size_passed = int(sys.argv[2])
elif len(sys.argv) == 4:
    ker_size = int(sys.argv[1])
    batch_size_passed = int(sys.argv[2])
    no_of_epochs = int(sys.argv[3])
elif len(sys.argv) == 5:
    ker_size = int(sys.argv[1])
    batch_size_passed = int(sys.argv[2])
    no_of_epochs = int(sys.argv[3])
    crp_count = int(sys.argv[4])
elif len(sys.argv) == 6:
    ker_size = int(sys.argv[1])
    batch_size_passed = int(sys.argv[2])
    no_of_epochs = int(sys.argv[3])
    crp_count = int(sys.argv[4])
    fc_count = int(sys.argv[5])

# Loading MNIST Dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Finding No. of Rows and Columns
rows_of_img = x_train[0].shape[0]
cols_of_img = x_train[1].shape[0]

# store the shape of a single image 
input_shape = (rows_of_img, cols_of_img, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Featuring Scaling - Normalization
x_train /= 255
x_test /= 255

# Doing One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

n_classes = y_test.shape[1]

# Set Kernel Size
kernel_size = (ker_size,ker_size)

# Creating model
model = Sequential()

# Adding CRP layers
model.add(Conv2D(20,kernel_size,padding="same",input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

count = 1
while count <= crp_count:
    model.add(Conv2D(50,kernel_size,padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    count+=1
    
# FC
model.add(Flatten())

count = 1
while count <= fc_count:
    model.add(Dense(500))
    model.add(Activation("relu"))
    count+=1
    
model.add(Dense(n_classes))
model.add(Activation("softmax")) 

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

print(model.summary())

# Conerting Images to 4D
x_train = x_train.reshape(x_train.shape[0], rows_of_img, cols_of_img, 1)
x_test = x_test.reshape(x_test.shape[0], rows_of_img, cols_of_img, 1)

# Training Parameters
batch_size = batch_size_passed
epochs = no_of_epochs

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), 
          shuffle=True)

model.save("mnist_LeNet.h5")   

# Evaluating the accuracy
scores = model.evaluate(x_test, y_test, verbose=1) 
print("\nAccuracy is :-\n") 
print(int(scores[1] * 100)) 
