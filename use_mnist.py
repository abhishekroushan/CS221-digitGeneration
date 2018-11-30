from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import matplotlib.image as mpimg
import pickle

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

img = pickle.load(open('custom2.pkl','rb'))
img = np.expand_dims(img,axis=0)
img = np.expand_dims(img,axis=3)
print(img.shape)

y = loaded_model.predict(img,verbose=1)
print(np.argmax(y))
