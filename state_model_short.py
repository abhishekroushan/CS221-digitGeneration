import numpy as np
import os
import collections
import util
import sys
import random
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
        self.actions=["up","down","right","left"]

    def startState(self):
        #state=((x,y) of robot ,canvas state)
        return (self.pen,[])
        

    def isEnd(self,state):
        #end when the coords reach 3/4th self.N
        curr_coords=state[0]
        return (curr_coords[0]>self.N*3/4) and (curr_coords[1]>self.N*3/4)

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
        trav_list=list(state[1])   #append later
        print(pen)
        curr_cost=len(trav_list)

        coords_list=trav_list.copy()
        coords_list.append(pen)
        for action in self.actions:
            bound_condn, new_pen=self.isBoundCoords(pen, action)
            new_state=(new_pen,coords_list)
            if self.isEnd(new_state): end_cost=end_state_cost(new_state)
            else: end_cost=0 

            if bound_condn:
                result.append((action, (new_pen,coords_list),curr_cost+end_cost))

        #print("result=",state,result)
        return result


def dynamicProgramming(problem):
    #cache=collections.defaultdict(int)
    cache ={}
    visited={}
    def futureCost(state):
        
        visited[str(state)]=True
        print("visited",state,visited.get(str(state)))
        def notvisited(s):
           print("notvisited",s, not visited.get(str(s)))
        #    if visited.get(str(s)) is True: flag=False
        #    else: flag=True
        #    print("flag",flag, not visited.get(str(s))) 
           return not visited.get(str(s))
            

        if problem.isEnd(state): return 0
        #print("state",state, type(state))
        fr_state=tuple(list(state))
        if fr_state in list(cache.keys()):
           print("state found in cache")
           return cache[fr_state]#python dict construct
        result=min((cost+futureCost(newState),action, newState, cost) for action, newState, cost in problem.getSuccStateAction(state) if notvisited(newState))
        cache[fr_state]=result
        return result
    state=problem.startState()
    #print("state=", state, type(state))
    totalCost=futureCost(state)
    history=[]
    while not problem.isEnd(state):
        _,action, newState, cost=cache[state]
        history.append((action, newState, cost))
        state=newState
    return (totalCost, history)

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



#testing
cz=28
init_canvas=np.zeros((cz,cz))
end_canvas=np.zeros((cz,cz))
end_canvas[2:6,4]=1
thresh=0.01
pen=[7,7]
model=StateModel(pen,cz)
ss=model.startState()
print("start_state", ss)
print("-----------------------")
test_pen=(7,7)
test_state=(test_pen,[(1,2),(7,7),(7,8),(20,16)])
print("test")
print(test_state)
print("-----------------------------------")
sa=model.getSuccStateAction(test_state)
print("result=",sa)
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

