import numpy as np
import matplotlib.pyplot as plt
import random

# actions correspond to index in [r,l,u,d] 
sz = 28

def getNextActionProbabilities(prev,prev2):
	# get next action using momentum of previous two strokes to increase their probabilities, leading to longer strokes and less re-tracing
	probs = np.array([0.25, 0.25, 0.25, 0.25])
	if prev == 0: no = 1
	elif prev == 1: no = 0
	elif prev ==2: no = 3
	elif prev == 3: no = 2
	eta = 1 
	probs[prev] += eta
	probs[prev2] += eta
	probs[no] = 0
	probs = probs/np.sum(probs)
	print("Probabilities: {}".format(probs))
	return probs

def nextAction(probs,pen):
	possible_actions = []
	if pen[0]<sz-1: possible_actions.append(0) #down
	if pen[1]>0: possible_actions.append(1) #left
	if pen[1]<sz-1: possible_actions.append(2) #right
	if pen[0]>0: possible_actions.append(3) #up
	action = -1
	while action not in possible_actions:
		rand = random.random()
		if rand<probs[0]: action=0
		elif rand<probs[0]+probs[1]: action=1
		elif rand<probs[0]+probs[1]+probs[2]: action=2
		else: action=3
	return action

canvas = np.zeros((28,28))
canvas[4,4] = 1
pen = [4,4]
prev = random.randint(0,4)
prev2 = random.randint(0,4)
num_moves = 0
while num_moves < 40:
	probs = getNextActionProbabilities(prev,prev2)
	action = nextAction(probs,pen)
	if action == 0: pen[0] += 1 #down
	elif action == 1: pen[1] -= 1 #left
	elif action == 2: pen[1] += 1 #right
	else: pen[0] -= 1 #up
	canvas[pen[0],pen[1]] = 1
	prev2 = prev
	prev = action
	plt.imshow(canvas)
	plt.pause(1)

