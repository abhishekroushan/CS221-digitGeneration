from keras.models import model_from_json
import numpy as np
from getkey import getkey, keys
import matplotlib.pyplot as plt
from keras.datasets import mnist
import skimage.morphology

#evaluate for getting mnist prediction for digit '2' of final result
def evaluate(img):
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=3)
    json_file = open('./old/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./old/model.h5")
    print("Loaded model from disk")
    y = loaded_model.predict(img,verbose=1)
    if np.argmax(y) == 2:
        prediction = print("Predicted 2 with confidence : {}".format(y[0,2]))
    return y[0,2]	



#load CNN_model from json
def get_model():
    with open("CNN_model.json","r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("CNN_model.h5")
    print("Loaded model architecture and weights")
    return loaded_model

#from a canvas of 1s and 0s, get the start and end coords of the bounding box which contains the digit fully
#used for getting start coordinates to place the pen
def get_start_coords(canvas):
    #canvas has 0,1s
    mask = canvas > 0
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    print("u",x0,"l",y0,"b", x1, "r",y1)
    for i in range(y0,y1):
        if canvas[x0,i]>0: break
    print("1st point at:",x0, i)
    return x0,i
    

#get samples of '2' from mnist database
def get_mnist_2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0   #in range 0,1
    print(x_train.shape)
    idx_2=np.where(y_train==2)
    x_train_2=x_train[idx_2]#training ex of img 2
    #rand_idx=59983
    #rand_idx for other cases of '2' in MNIST database
    rand_idx=np.random.choice(x_train_2.shape[0], 1)#choose 1 row
    #rand_idx for result in paper
    rand_idx=2226
    print("rand_idx",rand_idx)
    #single train ex for '2'
    x_train_2ex=x_train_2[int(rand_idx), :, :]

    #AIM: make the x_train_2ex as single streak of 0-1
    sharpen_2=skimage.morphology.skeletonize(x_train_2ex>0.5)
    sharpen_2.astype(int)
    canvas=np.zeros((28,28))
    canvas[sharpen_2==True]=1
    #plt.imshow(sharpen_2)
    #plt.pause(5)
    plt.imshow(x_train_2ex)
    plt.pause(5)
    return np.array(canvas)


def run_model(model,canvas):
    actions = []
    x = np.zeros((1,28,28,3))
    x[0,:,:,0] = canvas
    s0,s1=get_start_coords(canvas)
    #this pen start gets 98% confidence for '2'
    #pen = [5,14]
    #this pen start gets 93% confidence for '2'
    pen = [s0,s1]
    x[0,pen[0],pen[1],1:] = 1
    idx = 0
    while(sum(sum(x[0,:,:,1])) < sum(sum(canvas))):
        print(type(x))
        y = model.predict(x)
        plt.imshow(x[0,:,:,:])
        plt.pause(2)
        print('Predicted: {}'.format(y))
        action = np.argmax(y)
        idx += 1
        x[0,pen[0],pen[1],2] = 0
        if action == 0:
            actions.append('right')
            pen[1] += 1
        elif action == 1:
            actions.append('left')
            pen[1] -= 1
        elif action == 2:
            actions.append('up')
            pen[0] -= 1
        elif action == 3:
            actions.append('down')
            pen[0] += 1
        x[0,pen[0],pen[1],1:] = 1
    plt.imshow(x[0,:,:,1])
    plt.pause(10)
    evaluate(np.array(x[0,:,:,1]))

model = get_model()
canvas = get_mnist_2()
print("True")
#quit()
plt.imshow(canvas)
plt.pause(5)
run_model(model,canvas)
