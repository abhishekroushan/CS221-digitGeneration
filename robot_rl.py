import os
import numpy as np
import math, random
from collections import defaultdict
from keras.models import model_from_json
import pickle


#load mnist model
def load_mnist_model():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

#State action model class for defining the robotic mdp

class RoboticState(object):
    def __init__(self,pen,N, mnist_model):
        self.pen=pen
        self.N=N
        self.end=N*3/4
        self.actions=["up", "down", "left", "right"]
        self.action_probs=np.ones((self.N,self.N,len(self.actions)))*0.25 #for each pen state- action prob of 0.25 uniform to start with
        self.load_mnist_model=mnist_model
 
    def startState(self):
        #reset canvas
        self.canvas=np.zeros((self.N, self.N)) #state of the canvas from start state tracing out the actions- each time a state action pair is executed
        self.canvas[self.pen[0], self.pen[1]]=1 #update the canvas start state; 

        return (self.pen,None, None) #start state =tuple(pen_coords. prev_action, prev_prev_action)

    def isEnd(self, state):
        #end if the pen lies in 3/4th quadrant or any x,y ==N
        curr_pen=state[0]
        #end canvas
        is_canvas_end= curr_pen[0]<=0 or curr_pen[0]>=self.N-1 or curr_pen[1]<=0 or curr_pen[1]>=self.N-1
        is_canvas_lower_quad= curr_pen[0]>=self.end and curr_pen[1]>=self.end
        if is_canvas_end: return True
        return is_canvas_lower_quad       

    def isBounded(self, pen):
        return (pen[0]>=0 and pen[0]<self.N) and (pen[1]>=0 and pen[1]<self.N) #(0,0)<=(x,y) <(N,N) 

    def evaluate(self,img):
        loaded_model=self.load_mnist_model
        img = np.expand_dims(img,axis=0)
        img = np.expand_dims(img,axis=3)
        y = loaded_model.predict(img,verbose=1)
        if np.argmax(y) == 2:
            prediction = print("Predicted 2 with confidence : {}".format(y[0,2]))
        print("img score=", y[0,2])
        return y[0,2]

    def adjust_probs(self,state):
    #action_prob=prob of actions as a list
    #idx 1,2= index of prev, prev_prev actions
        curr_pen, prev_action, prev_prev_action= state #unpack
        eta=1
        action_prob_copy=(self.action_probs[curr_pen[0], curr_pen[1]]).copy() #in range 0,1
        if prev_action!=None:
            prev_action_idx= (self.actions).index(prev_action)
            action_prob_copy[prev_action_idx]+=eta

        if prev_prev_action!=None:
            prev_prev_action_idx=(self.actions).index(prev_prev_action)
            action_prob_copy[prev_prev_action_idx]+=eta
        
        action_prob_copy/=sum(action_prob_copy)
        #if idx1==None: return action_prob_copy

        return action_prob_copy



    def succStateAction(self, state, action):
        #action here is the chosen one 
        #given state and action, return (new_pen_coords, curr_action, prev_action)
        #update self.canvas for the state, action
        curr_pen, prev_action, prev_prev_action= state #unpack
        new_pen=list(curr_pen)
        if action=="up": new_pen[0]-=1
        if action=="down": new_pen[0]+=1
        if action=="left": new_pen[1]-=1
        if action=="right": new_pen[1]+=1
        #update canvas
        if self.isBounded(new_pen): self.canvas[new_pen[0], new_pen[1]]=1

        return (tuple(new_pen), action, prev_action)


def rl_algorithm():
    #problem is a state space object
    eta=5
    sz=28
    start_pen=(7,7)  
    mnist_model=load_mnist_model()  

    #init
    problem=RoboticState(start_pen, sz, mnist_model)

    for i in range(50):

        startState=problem.startState()
        currState=startState
        state_action_list=[]
        while not problem.isEnd(currState):
            pen, prev_action, prev_prev_action=currState #unpack
            #print("action_probs before adjusting:", problem.action_probs[pen[0], pen[1]])
            action_probs=problem.adjust_probs(currState)#adjust probs for prev_action and prev_prev_action
            #print("state =", currState, "action_probs=", action_probs)
            action=random.choices(population=problem.actions, weights=action_probs, k=1)[0] #ex action="up"
            state_action_list.append((pen, action))
            succState=problem.succStateAction(currState, action)
            currState=succState

        #reached end
        #update reward and problem.action_probs
        score=problem.evaluate(problem.canvas)
        for items in state_action_list:
            p, a= items
            pen_action_prob=problem.action_probs[p[0], p[1]]
            #print("pen_action_prob before adjust=",pen_action_prob)
            action_idx=(problem.actions).index(a)
            
            if score>0.5:
            #high pred, high reward
                adjust=np.zeros(len(problem.actions))
                adjust[action_idx]=score

            if score<0.5:
                adjust=np.ones(len(problem.actions))
                adjust[action_idx]= 0

            pen_action_prob+=adjust
            
            pen_action_prob/=sum(pen_action_prob)
            problem.action_probs[p[0], p[1]]=pen_action_prob


    pickle.dump(problem.action_probs, open('weights','wb'))
    #inference
    print("inference-------------------")
    startState=problem.startState()
    currState=startState
    state_action_list=[]
    while not problem.isEnd(currState):
        pen, prev_action, prev_prev_action=currState #unpack
        action_probs=problem.action_probs[pen[0], pen[1]]#adjust probs for prev_action and prev_prev_action

        action=random.choices(population=problem.actions, weights=action_probs, k=1)[0] #ex action="up"
        state_action_list.append((pen, action))
        succState=problem.succStateAction(currState, action)
        currState=succState
    score=problem.evaluate(problem.canvas)

    



#functions to check the states
print("-------------------------------")
#sz=28
#start_pen=(7,7)  
#mnist_model=load_mnist_model()  
#init
#problem=RoboticState(start_pen, sz, mnist_model)
#ss=problem.startState()
#print("startState=", ss)
#ns=problem.succStateAction(ss, "up")
#print("nextState=", ns)
#print("is ns end= ", problem.isEnd(ns))

print("-------------------------------")
print("rl_algorithm:")
rl_algorithm()
