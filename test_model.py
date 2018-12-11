from keras.models import model_from_json
import numpy as np
from getkey import getkey, keys
import matplotlib.pyplot as plt

def get_model():
    with open("CNN_model.json","r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("CNN_model.h5")
    print("Loaded model architecture and weights")
    return loaded_model

def get_artificial_2():
    actions = []
    pen = [4,4]
    canvas = np.zeros((28,28))
    canvas[pen[0],pen[1]] = 1
    plt.imshow(canvas)
    plt.pause(0.1)
    while(True):
        print("Press U,D,L,R")
        key = getkey()
        if key == keys.UP:
            actions.append('up')
            pen[0] -= 1
        elif key == keys.DOWN:
            actions.append('down')
            pen[0] += 1
        elif key == keys.LEFT:
            actions.append('left')
            pen[1] -= 1
        elif key == keys.RIGHT:
            actions.append('right')
            pen[1] += 1
        else:
            break
        canvas[pen[0],pen[1]] = 1
        plt.imshow(canvas)
        plt.pause(0.1)
    return canvas, actions

def run_model(model,canvas,corr_actions):
    actions = []
    x = np.zeros((1,28,28,3))
    x[0,:,:,0] = canvas
    pen = [4,4]
    x[0,pen[0],pen[1],1:] = 1
    idx = 0
    while(sum(sum(x[0,:,:,1])) < sum(sum(canvas))):
        print(type(x))
        y = model.predict(x)
        plt.imshow(x[0,:,:,:])
        plt.pause(2)
        print('Predicted: {}'.format(y))
        action = np.argmax(y)
        print('True: {}'.format(corr_actions[idx]))
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

model = get_model()
canvas, actions = get_artificial_2()
print("True")
plt.imshow(canvas)
plt.pause(5)
print("")
print(actions)
run_model(model,canvas,actions)
