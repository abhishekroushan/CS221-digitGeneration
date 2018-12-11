from keras.models import model_from_json
import numpy as np
import matplotlib.image as mpimg
import pickle

def evaluate(img):
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	y = loaded_model.predict(img,verbose=1)
	if np.argmax(y) == 2:
		prediction = print("Predicted 2 with confidence : {}".format(y[0,2]))
	return y[0,2]	

img = pickle.load(open('custom2.pkl','rb'))
img = np.expand_dims(img,axis=0)
img = np.expand_dims(img,axis=3)
print(img.shape)
evaluate(img)
