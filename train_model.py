'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Input
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

batch_size = 1
num_classes = 4
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

def load_data():
    x_train=np.load('x_train.npy')
    y_train=np.load('y_train.npy')
    x_test=np.load('x_test.npy')
    y_test=np.load('y_test.npy')
    return x_train, y_train, x_test, y_test

# get and format the data, split between train and test sets
x_train, y_train, x_test, y_test=load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y train/test after to_categorical")
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

def action_CNN(input_shape, nclasses, data_format, act='relu', init='glorot_uniform'):
  input_img=Input(shape=input_shape)#3x28x28 ex
  print("input:", input_img.shape)
  x = Conv2D(32, (3, 3), activation='relu', padding='same',data_format=data_format, name='conv1')(input_img)
  x = MaxPooling2D((2, 2), padding='same',data_format=data_format,name='maxpool1')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same',data_format=data_format,name='conv2')(x)
  x = MaxPooling2D((2, 2), padding='same',data_format=data_format, name='maxpool2')(x)
  x = Dropout(0.25)(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same',data_format=data_format,name='conv3')(x)
  enc = MaxPooling2D((2, 2), padding='same',data_format=data_format,name='maxpool3')(x)
  lin= Flatten()(enc)
  logits=Dense(nclasses,activation='linear',name='dense1')(lin)
  probs=Activation('softmax')(logits)
  return Model(inputs=input_img,outputs=probs, name='action_model')

#test summary model
def create_model():
    xtrain, ytrain, xtest, ytest=x_train, y_train, x_test, y_test
    my_model=action_CNN((28,28,3), num_classes, 'channels_last')
    my_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    my_model.summary()
    my_model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))
    score=my_model.evaluate(xtest,ytest, verbose=0)
    model_json = my_model.to_json()
    with open("CNN_model_20.json","w") as json_file:
        json_file.write(model_json)
    my_model.save_weights("CNN_model_20.h5")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return my_model

def get_model():
    with open("CNN_model.json","r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("CNN_model.h5")
    print("Loaded model architecture and weights")
    return loaded_model

my_model = create_model()
my_model = get_model()

#visualize
for i in range(0,7000):
    x_example = np.zeros((1,28,28,3))
    x_example[0,:,:,:] = x_train[i,:,:,:]
    print(type(x_example))
    y_pr = my_model.predict(x_example)
    print('Predicted: {}'.format(np.argmax(y_pr)))
    y_true = y_train[i]
    print('True: {}'.format(np.argmax(y_true)))
    plt.imshow(x_train[i,:,:,:])
    plt.pause(1)

