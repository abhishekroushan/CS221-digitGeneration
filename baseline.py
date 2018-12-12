'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import numpy as np
import matplotlib.pyplot as plt
import random

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

def guess_action(canvas,pen):
    actions = []
    if pen[0]>0 and canvas[pen[0]-1,pen[1]] == 1: actions.append('up')
    if pen[0]<27 and canvas[pen[0]+1,pen[1]] == 1: actions.append('down')
    if pen[1]>0 and canvas[pen[0],pen[1]-1] == 1: actions.append('left')
    if pen[1]<27 and canvas[pen[0],pen[1]+1] == 1: actions.append('right')
    print(actions)
    r = random.random()*len(actions)
    print(r)
    print(int(np.floor(r)))
    return actions[int(np.floor(r))]

for idx in range(0,1000):
    example = np.zeros((28,28,3))
    example[:,:,0] = x_train[idx,:,:,0]
    example[:,:,1] = x_train[idx,:,:,2]
    example[:,:,2] = x_train[idx,:,:,2]
    pen = x_train[idx,:,:,2]
    pen_x = np.argmax(np.sum(pen,axis=1))
    pen_y = np.argmax(np.sum(pen,axis=0))
    guess = guess_action(example[:,:,0],[pen_x,pen_y])
    true = y_train[idx]
    if true == 0: true_action = 'right'
    elif true == 1: true_action = 'left'
    elif true == 2: true_action = 'up'
    else: true_action = 'down'
    print('X,Y: {},{}'.format(pen_x,pen_y))
    print('Guess: {}'.format(guess))
    print('True: {}'.format(true_action))
    plt.imshow(example)
    plt.pause(15)

