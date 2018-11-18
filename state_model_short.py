import numpy as np
import os
import collections
import util
import sys
import random
import util_ext
import heapq
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)


def end_state_cost(state):
    prev_coords = state[1]
    expected_img = np.zeros((28,28))
    expected_img[6:9,6:18] = 1
    expected_img[9:11,6:10] = 1
    expected_img[9:12,15:20] = 1
    expected_img[12:20,17:21] = 1
    expected_img[18:22,6:18] = 1
    expected_img[20:23,16:22] = 1
    expected_img[20:23,6:10] = 1
    #plt.imshow(expected_img)
    #plt.pause(5)
    reward = 0
    for x,y in prev_coords:
        if expected_img[x,y] == 1:
            reward += 1
    return reward

#print(end_state_cost(((7,7),[(1,2),(7,7),(7,8),(20,16)])))

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
        return (curr_coords[0]>self.end) and (curr_coords[1]>self.end)

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
        
        curr_cost=1

        for action in self.actions:
            bound_condn, new_pen=self.isBoundCoords(pen, action)
            new_state=(new_pen,pen,prev_pen)
            if bound_condn:
                if self.isEnd(new_state): 
                    curr_cost += end_state_cost(new_state)
                result.append((action, new_state, curr_cost))

        #print("result=",state,result)
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
    print("startState",startState, type(startState))
    frontier.put(startState,0)
    came_from={}
    cost_so_far={}
    came_from[startState]=None
    cost_so_far[startState]=0

    while not frontier.empty():
        current=frontier.get()
        if problem.isEnd(current): break
        for action, newState, cost in problem.getSuccStateAction(current):
            new_cost=cost_so_far[current]+cost
            if newState not in cost_so_far or new_cost<cost_so_far[newState]:
                cost_so_far[newState]=new_cost
                #need to pass in current_pen_coords, end_pen_coords to heuristic()
                priority=new_cost+heuristic(newState[0],(problem.end,problem.end))
                frontier.put(newState,priority)
                came_from[newState]=current

    return came_from, cost_so_far



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
#print((sa[1])[2])
#ea=model.isEnd((sa[1])[1])
#print("isend=",ea)
#for item in sa:
#    print(item[0])
#    print(item[1])
#    print(item[2])
#    print("-----------------------")
#
#printSolution(dynamicProgramming(model))
#printSolution(uniformCostSearch(model))
print(a_star(model))