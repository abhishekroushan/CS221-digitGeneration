from keras.models import model_from_json
import random
import numpy as np
import matplotlib.image as mpimg
import pickle
import matplotlib.pyplot as plt

# Loads MNIST-trained CNN to classify digit in a (28,28) canvas 
def load_json():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	return loaded_model	

# Uses MNIST-trained CNN to classify digit in a (28,28) canvas and return cost relative to its similarity to a 2
def evaluate(img):
	img = np.expand_dims(img,axis=0)
	img = np.expand_dims(img,axis=3)
	y = loaded_model.predict(img,verbose=1)
	print(y)
	prob2 =  y[0,2]	
	reward = prob2-0.2	
	return reward

# Test the image that Clara custom-made
def test_oracle():
	img = pickle.load(open('custom2.pkl','rb'))
	print(img.shape)
	cost = evaluate(img)
	print(cost)


class RLmodel(object):
	def __init__(self,pen,N):
		self.startpen=pen          #(x,y) tuple robot start state
		self.N=N                   #canvas size
		self.end=N*3/4.0	   #threshold for lower-right-hand part of canvas
		self.p_inde = {}	   #probabilities of each action independent of previous actions
		for x in range(0,N):	   #initially all probabilities are equal
			for y in range(0,N):
				self.p_inde[(x,y)] = (0.25, 0.25, 0.25, 0.25)
		self.actions=["r","l","u","d"] #possible actions from any state

	def startState(self):
        	#state=((x,y),prev,prev_prev)
        	return (self.startpen,None,None)

	def isEnd(self,state):
        	#end when the coords reach 3/4th self.N or go off canvas
		curr_coords=state[0]
		if curr_coords[0]<=0 or curr_coords[0]>=self.N-1 or curr_coords[1]<=0 or curr_coords[1]>=self.N-1: return True
		return ((curr_coords[0]>=self.end) and (curr_coords[1]>=self.end))

	def getActions(self,state):
		# handle edge cases in isEnd
		return self.actions

	def getSuccessor(self,state,a):
		pen, prev, prev_prev = state
		if a == "r": new_pen = (pen[0],pen[1]+1)
		elif a == "l": new_pen = (pen[0],pen[1]-1)
		elif a == "d": new_pen = (pen[0]+1,pen[1])
		elif a == "u": new_pen = (pen[0]-1,pen[1])
		else: print("Invalid action")
		return (new_pen,a,prev)

	def getIndex(self,action):
		# get index of a given action in self.actions
		for i in range(0,4):
			if action == self.actions[i]: return i
		return -1

	def getAdjustedProbs(self,state):
		# get adjusted probabilities, using momentum of previous two strokes to increase their probabilities
		p_inde = self.p_inde[state[0]]
		prev = self.getIndex(state[1])
		prev_prev = self.getIndex(state[2])
		eta = 2
		aug = np.zeros(4)
		for i in range(0,4):
			aug[i] = eta*(prev == i)+eta*(prev_prev == i)
		augtot = np.sum(aug)+1
		pr = (p_inde[0]+aug[0])/augtot
		p = ((p_inde[0]+aug[0])/augtot, (p_inde[1]+aug[1])/augtot,(p_inde[2]+aug[2])/augtot, (p_inde[3]+aug[3])/augtot)
		return p

	def makeCanvas(self,actions,view):
		canvas = np.zeros((self.N,self.N))
		pen = [self.startpen[0], self.startpen[1]]
		for a in actions:
			canvas[pen[0],pen[1]] = 1
			if a == "r": pen[1] += 1
			elif a == "l": pen[1] -= 1
			elif a == "d": pen[0] += 1
			elif a == "u": pen[0] -= 1
			else: print("Invalid action in actions")
		return canvas

def nextAction(probs,actions):
	rand = random.random()
	if rand<probs[0]: return actions[0]
	elif rand<probs[0]+probs[1]: return actions[1]
	elif rand<probs[0]+probs[1]+probs[2]: return actions[2]
	else: return actions[3]

# TESTING EACH METHOD
# print(model.getSuccessor(state,'u'))
# print("Start state: {}".format(ourmodel.startState()))
# print("Should be end: {}".format(ourmodel.isEnd(((22,25),'right','right'))))
# print("Should not be end: {}".format(ourmodel.isEnd(ourmodel.startState())))
# print("Adjusted probs after moving right, down: {}".format(ourmodel.getAdjustedProbs(((7,9),'r','r'))))
# ourmodel.makeCanvas(['r','r','r','d','d','d'])

pen_start = (7,7)
sz = 28
loaded_model = load_json()
model = RLmodel(pen_start,sz)
eta = 0.2

for iteration in range(0,10000):
	# initialize
	state = model.startState()
	num_moves = 0
	actions = []
	
	# one iteration
	while(num_moves < 100):
		num_moves += 1
		probs = model.getAdjustedProbs(state)
		possible_actions = model.getActions(state)
		next_action = nextAction(probs,possible_actions)
		state = model.getSuccessor(state,next_action) 
		if(model.isEnd(state)): break
		actions.append(next_action)
	
	# learning
	canvas = model.makeCanvas(actions,True)
	reward = evaluate(canvas)
	print("Reward: {}".format(reward))
	state = model.startState()
	plt.imshow(canvas)
	plt.pause(2)	
	for action in actions:
		pen_loc = state[0]
		action_idx = model.getIndex(action)
		old_probs = model.p_inde[pen_loc]
		if reward > 0:
			adjusts = np.zeros(4)
			adjusts[action_idx] = reward
			denom = 1+reward
		else:
			adjusts = -reward*np.ones(4)
			adjusts[action_idx] = 0
			denom = 1+(-reward*3)
		new_probs = np.ones(4)*0.25
		for i in range(0,4):
			new_probs[i] = (old_probs[i]+adjusts[i])/denom
		assert(np.abs(np.sum(new_probs)-1) < 0.0001)
		model.p_inde[pen_loc] = (new_probs[0],new_probs[1],new_probs[2],new_probs[3])
		state = model.getSuccessor(state,action)
		
	
