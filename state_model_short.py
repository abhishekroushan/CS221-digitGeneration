import numpy as np
import os
import collections
import util
import sys
import random
import util_ext
import heapq
from visualization import visualize
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)


def end_state_cost(state):
    expected_img = np.zeros((28,28))
    expected_img[6:9,6:18] = 1
    expected_img[9:11,6:10] = 1
    expected_img[9:12,15:20] = 1
    expected_img[12:20,17:21] = 1
    expected_img[18:22,6:18] = 1
    expected_img[20:23,16:24] = 1
    expected_img[20:23,6:10] = 1
    #plt.imshow(expected_img)
    #plt.pause(5) 
    cost = 1
    for i in range(0,3):
        x = state[i][0]
        y = state[i][1]
        if expected_img[x,y] == 0:
            cost += 2
    if state[0][0] == state[2][0] and state[0][1] == state[2][1]:
        cost += 10
    return cost

print(end_state_cost(((7,7),(7,8),(20,16))))

#GUIDE:
#   startState=(pen_position in row,col , list of all pen coords traversed till now)
#   isEnd= when the pen coords cross 3/4th of the canvas size
#   getSuccStateAction= for all actions in self.actions: return (action, newState, cost given state,action)

class StateModel(object):
    def __init__(self,pen,N):
        self.pen=pen                #(x,y) tuple robot start state
        self.N=N                    #canvas size
        self.end=N*3/4
        self.actions=["up","down","right","left"]

    def startState(self):
        #state=((x,y) of robot ,canvas state)
        return (self.pen,self.pen,self.pen)
        

    def isEnd(self,state):
        #end when the coords reach 3/4th self.N
        curr_coords=state[0]
        return (curr_coords[0]>=self.end) and (curr_coords[1]>=self.end)

    def isBoundCoords(self,curr_pen, action):
        #pen x,y <=N
        #print("pos=",curr_pos[0]<self.N and curr_pos[1]<self.N)
        new_pen=list(curr_pen[:])
        if action=="up": new_pen[0]-=1
        if action=="down": new_pen[0]+=1
        if action=="left":new_pen[1]-=1
        if action=="right":new_pen[1]+=1
        return ((new_pen[0]<self.N and new_pen[0]>=0) and (new_pen[1]<self.N and new_pen[1]>=0)),tuple(new_pen)


    def getSuccStateAction(self,state):
        #current_state:action->next_state
        #action=white pixel vicinity, up:down:center:left
        result=[]
        
        pen=state[0][:] #make a copy
        prev_pen = state[1]
        
        for action in self.actions:
            bound_condn, new_pen=self.isBoundCoords(pen, action)
            new_state=(new_pen,pen,prev_pen)
            if bound_condn:
                action_cost = end_state_cost(new_state)
                result.append((action, new_state, action_cost))

        return result



def uniformCostSearch(problem):
    frontier=util.PriorityQueue()
    frontier.update(problem.startState(),0)
    while True:
        state, pastCost=frontier.removeMin()
        if problem.isEnd(state):
            return (pastCost,[])#no history
        for action, newState, cost in problem.getSuccStateAction(state):
            frontier.update(newState, pastCost+cost)

def printSolution(solution):
    totalCost, history = solution
    print ("totalCost:", totalCost)
    for item in history:
        print (item)

def heuristic(curr,end):
    #return abs coords difference.
    (x1,y1)=curr
    (x2,y2)=end
    return abs(x1-x2)+abs(y1-y2)

def a_star(problem):
    frontier=util_ext.PriorityQueue()
    startState=problem.startState()
    initial = ([],startState,0)
    future_cost = heuristic(startState[0],(problem.end,problem.end))
    print(future_cost)
    frontier.put(initial,future_cost)
    counter = 0
    while not frontier.empty(): # and counter < 500:
        counter += 1
        current=frontier.get()
        #print("current")
        #print(current)
        current_actions = current[0]
        current_state = current[1]
        current_cost = current[2]
        # if current_state[0][0] >= 21 and current_state[0][1] >= 21: return current
        if problem.isEnd(current_state): return current
        for action, newState, cost in problem.getSuccStateAction(current_state):
            next_cost=current_cost+cost
            next_actions = current_actions + [action]
            next_all = (next_actions,newState,next_cost)
            #print("put")
            #print(next_all)
            future_cost = heuristic(newState[0],(problem.end,problem.end))
            #print(next_cost+future_cost)
            frontier.put(next_all,next_cost+future_cost)
    return None



#testing
sz=28
start_pen=(7,7)
model=StateModel(start_pen,sz)
ss=model.startState()
test_state = ss

for i in range(0,-1):
    print("-----------------------")
    retval = model.getSuccStateAction(test_state)
    test_state = retval[0][1]
    print(retval)

soln = a_star(model)

if soln is not None:
    print("Found solution!")
    print(soln)

soln_actions=soln[0]
visualize(start_pen, soln_actions,sz)


