import numpy as np
import os
import collections
import util
import sys
import random
sys.setrecursionlimit(10000)


#GUIDE:
#   startState=(pen_position in row,col , initial_state where the canvas is all zeros ie Black)
#   isEnd= if the current_canvas state differ with end state by threshold
#   newState= new_position and state corresponding given action:
#       #changes pen coordinates
#       #cahnges canvas state with cell corresp to pen_coords=1 ie White
#   getSuccStateAction= for all actions in self.actions: return (state, action, newState given state,action)

class StateModel(object):
    def __init__(self, init, end, pen, threshold,N):
        self.init=init              #initial canvas state--ideally should be all zeros
        self.end=end                #target canvas state
        self.threshold=threshold    #thresh start and end state
        self.pen=pen                #[x,y] robot
        self.N=N                    #canvas size
        self.actions=["up","down","right","left"]

    def startState(self):
        #state=((x,y) of robot ,canvas state)
        return (self.pen,self.init)

    def isEnd(self,state):
        #norm state-end<=threshold
        canvas_state=np.copy(state[1])
        return np.linalg.norm(canvas_state-self.end)<=self.threshold

    def isBound(self,curr_pen, action):
        #pen x,y <=N
        #print("pos=",curr_pos[0]<self.N and curr_pos[1]<self.N)
        new_pen=curr_pen[:]
        if action=="up": new_pen[0]-=1
        if action=="down": new_pen[0]+=1
        if action=="left":new_pen[1]-=1
        if action=="right":new_pen[1]+=1
        return ((new_pen[0]<self.N and new_pen[0]>=0) and (new_pen[1]<self.N and new_pen[1]>=0))

    def newStateCost(self, state,action):

        #print("pen_states=",pen, self.isBound(pen))
        p=state[0][:]
        canvas_state=np.copy(state[1])
        if action=="up": p[0]-=1
        if action=="down": p[0]+=1
        if action=="left":p[1]-=1
        if action=="right":p[1]+=1
        canvas_state[p[0],p[1]]=1
        cost=np.linalg.norm(canvas_state*self.end)
        return (p, canvas_state),cost

    def getSuccStateAction(self,state):
        #current_state:action->next_state
        #action=white pixel vicinity, up:down:center:left
        result=[]
        
        pen=state[0][:]
        if self.isBound(pen,"up"):
            new_state, cost=self.newStateCost(state, "up")
            result.append(("up",new_state, cost))
        if self.isBound(pen,"right"):
            new_state, cost=self.newStateCost(state, "right")
            result.append(("right",new_state, cost))
        if self.isBound(pen,"left"):
            new_state, cost=self.newStateCost(state, "left")
            result.append(("left",new_state, cost))
        if self.isBound(pen,"down"):
            new_state, cost=self.newStateCost(state, "down")
            result.append(("down",new_state, cost))


        #print("result=",state,result)
        return result


def dynamicProgramming(problem):
    #cache=collections.defaultdict(int)
    cache ={}
    def futureCost(state):
        #visited={}
        #visited[str(state)]=True
        #print(state,visited.values())
        #def notvisited(s):
        #    print("notvisited",s,visited.get(str(s)))
        #    if visited.get(str(s)) is True: flag=False
        #    else: flag=True
        #    print("flag",flag, not visited.get(str(s))) 
        #   return flag
            

        if problem.isEnd(state): return 0
        #print("state",state, type(state))
        fr_state=tuple(list(state))
        if fr_state in list(cache.keys()):
           print("state found in cache")
           return cache[fr_state]#python dict construct
        result=min((cost+futureCost(newState),action, newState, cost) for action, newState, cost in problem.getSuccStateAction(state))
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
cz=8
init_canvas=np.zeros((cz,cz))
end_canvas=np.zeros((cz,cz))
end_canvas[2:6,4]=1
thresh=0.01
pen=[2,4]
model=StateModel(init_canvas,end_canvas,pen,thresh,cz)
model.startState()
print("-----------------------")
test_canvas=np.copy(end_canvas)
test_canvas[5:6,4]=0
test_pen=[4,4]
test_state=(test_pen,test_canvas)
print("test")
print(test_state)
print("-----------------------------------")
sa=model.getSuccStateAction(test_state)
print((sa[1])[2])
ea=model.isEnd((sa[1])[1])
print("isend=",ea)
#for item in sa:
#    print(item[0])
#    print(item[1])
#    print(item[2])
#    print("-----------------------")
#
printSolution(dynamicProgramming(model))
#printSolution(uniformCostSearch(model))

