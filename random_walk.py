import random
import numpy as np
import matplotlib.pyplot as plt

for ex in range(0,10):
    canvas = np.zeros((28,28))
    pen_x = 4
    pen_y = 4
    for move in range(0,70):
        canvas[pen_x,pen_y] = 1
        actions = []
        if pen_x < 27: actions.append('down')
        if pen_x > 0: actions.append('up')
        if pen_y < 27: actions.append('right')
        if pen_y >0: actions.append('left')
        choice = actions[int(np.floor(random.random()*len(actions)))]
        if choice == 'down': pen_x += 1
        elif choice == 'up': pen_x -= 1
        elif choice == 'right': pen_y += 1
        elif choice == 'left': pen_y -= 1
    plt.imshow(canvas)
    plt.pause(8)
