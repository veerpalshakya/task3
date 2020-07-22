from keras.datasets import mnist
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

img_rows = x_train[0].shape[0]
img_cols = x_train[0].shape[1]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Convolution2D(filters=5, kernel_size=(5,5), activation='relu', input_shape=input_shape  ))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(filters=10, kernel_size=(5,5), activation='relu'  ))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))
           
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
print(model.summary())
Trained_model = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test) ,)
model.save("mnist.h5")

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

acc=Trained_model.history['accuracy'][-1]*100
print('accuracy:',acc)
int_acc=int(acc)
f=open("model_accuracy.txt",'w')
f.write("{}".format(int_acc))
f.close()
