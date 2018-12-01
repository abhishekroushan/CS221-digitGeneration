from keras.models import model_from_json
import numpy as np
import matplotlib.image as mpimg
import pickle
import matplotlib.pyplot as plt

def load_json():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	return loaded_model	

def evaluate(img):
	img = np.expand_dims(img,axis=0)
	img = np.expand_dims(img,axis=3)
	y = loaded_model.predict(img,verbose=1)
	print(y)
	prob2 =  y[0,2]	
	cost = -100*(prob2-0.5)	
	return cost

def test_oracle():
	img = pickle.load(open('custom2.pkl','rb'))
	print(img.shape)
	cost = evaluate(img)
	print(cost)

class RLmodel(object):
	def __init__(self,pen,N):
		self.startpen=pen                #(x,y) tuple robot start state
		self.N=N                    #canvas size
		self.end=N*3/4.0
		self.p_inde = {}
		for x in range(0,N):
			for y in range(0,N):
				self.p_inde[(x,y)] = (0.25, 0.25, 0.25, 0.25)
		self.actions=["r","l","u","d"]

	def startState(self):
        	#state=((x,y),prev,prev_prev)
        	return (self.startpen,None,None)

	def isEnd(self,state):
        	#end when the coords reach 3/4th self.N or go off canvas
		curr_coords=state[0]
		if curr_coords[0]<=0 or curr_coords[0]>=self.N-1 or curr_coords[1]<=0 or curr_coords[1]>=self.N-1: return True
		return ((curr_coords[0]>=self.end) and (curr_coords[1]>=self.end))

	def getActions(self,state):
		return self.actions

	def getIndex(self,action):
		for i in range(0,4):
			if action == self.actions[i]: return i
		return -1

	def getAdjustedProbs(self,state):
		prev = getIndex(state[1])
		prev_prev = getIndex(state[2])
		eta = 0.5
		aug = np.zeros(4)
		for i in range(0,4):
			aug[i] = eta*(prev == i)+eta*(prev_prev == i)
		augtot = np.sum(aug)
		p = ((self.p_inde[0]+aug[0])/augtot, (self.p_inde[1]+aug[1])/augtot,(self.p_inde[2]+aug[2])/augtot, (self.p_inde[3]+aug[3])/augtot)
		return p

	def makeCanvas(self,actions):
		canvas = np.zeros((self.N,self.N))
		pen = [self.startpen[0], self.startpen[1]]
		for a in actions:
			canvas[pen[0],pen[1]] = 1
			if a == "r": pen[1] += 1
			elif a == "l": pen[1] -= 1
			elif a == "d": pen[0] += 1
			elif a == "u": pen[0] -= 1
			else: print("Invalid action in actions")
		plt.imshow(canvas)
		plt.pause(4)	
		evaluate(canvas)

loaded_model = load_json()
ourmodel = RLmodel((7,7),28)
print("Start state: {}".format(ourmodel.startState()))
print("Should be end: {}".format(ourmodel.isEnd(((22,25),'right','right'))))
print("Should not be end: {}".format(ourmodel.isEnd(ourmodel.startState())))
ourmodel.makeCanvas(['r','r','r','d','d','d'])


