import numpy as np
#import matplotlib.pyplot as plt
import random

# actions correspond to index in [r,l,u,d] 
sz = 28

def getNextActionProbabilities(prev,prev2,prev3):
    # get next action using momentum of previous two strokes to increase their probabilities, leading to longer strokes and less re-tracing
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    if prev == 0: no = 1
    elif prev == 1: no = 0
    elif prev ==2: no = 3
    elif prev == 3: no = 2
    eta = 1.5
    probs[prev] += eta
    probs[prev2] += eta
    probs[prev3] += eta
    probs[no] = 0
    probs = probs/np.sum(probs)
    return probs

def nextAction(probs,pen):
    possible_actions = []
    if pen[1]<sz-2: possible_actions.append(0) #right
    if pen[1]>1: possible_actions.append(1) #left
    if pen[0]>1: possible_actions.append(2) #up
    if pen[0]<sz-2: possible_actions.append(3) #down
    action = -1
    while action not in possible_actions:
        rand = random.random()
        if rand<probs[0]: action=0
        elif rand<probs[0]+probs[1]: action=1
        elif rand<probs[0]+probs[1]+probs[2]: action=2
        else: action=3
    return action
max_moves = 70
n_examples = 1000
x = np.zeros((70000,28,28,3))#dyn inc #batches
y = np.zeros((70000))#dyn inc #batches
for n in range(0,n_examples):
    canvas = np.zeros((28,28))
    canvas[4,4] = 1
    pen = [4,4]
    prev = random.randint(0,3)
    prev2 = random.randint(0,3)
    prev3 = random.randint(0,3)
    num_moves = 0
    while num_moves < max_moves:
        probs = getNextActionProbabilities(prev,prev2,prev3)
        action = nextAction(probs,pen)
        #print("action: {}".format(action))
        if action == 0: pen[1] += 1 #right
        elif action == 1: pen[1] -= 1 #left
        elif action == 2: pen[0] -= 1 #up
        else: pen[0] += 1 #down
        canvas[pen[0],pen[1]] = 1
        print(n*max_moves+num_moves)
        x[n*max_moves+num_moves,:,:,1]=canvas
        x[n*max_moves+num_moves,pen[0],pen[1],2]=1
        y[n*max_moves+num_moves]=action
        prev3 = prev2
        prev2 = prev
        prev = action
        num_moves += 1
    
    x[n*num_moves:(n+1)*num_moves,:,:,0]=canvas
    #plt.imshow(canvas)
    #plt.pause(1)
#random split
x_train=np.array(x[0:60000, :,:,:])
print("x_train.shape", x_train.shape)
x_test=np.array(x[60001:70000, :,:,:])
print("x_test.shape", x_test.shape)
y_train=np.array(y[0:60000])
print("y_train.shape", y_train.shape)
y_test=np.array(y[60001:70000])
print("y_test.shape", y_test.shape)

np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
#print(c2.shape)
##number of zeros
#print("zero batches")
#true_lst=[]
#for i in range(70000):
#    if not np.any(c2[i, :,:,:]): true_list.append(True)
#print(sum(true_lst))
##print(sum[(not np.any(c2[i,:,:,:]) for i in range(70000))])
