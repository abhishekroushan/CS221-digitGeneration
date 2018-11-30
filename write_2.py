import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import pickle

def write_2():
    expected_img = np.zeros((28,28))
    expected_img[6:9,6:18] = 1
    expected_img[9:11,6:10] = 1
    expected_img[9:12,15:20] = 1
    expected_img[12:20,17:21] = 1
    expected_img[18:22,6:18] = 1
    expected_img[20:23,16:24] = 1
    expected_img[20:23,6:10] = 1
    pickle.dump(expected_img,open('custom2.pkl','wb'))
    plt.imshow(expected_img)
    plt.pause(5) 

write_2()
