import numpy as np
import os

#GUIDE:
#   startState=(pen_position in row,col , initial_state where the canvas is all zeros ie Black)
#   isEnd= if the current_canvas state differ with end state by threshold
#   newPosition= new_position and state corresponding given action:
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
        self.actions=['up','down','right','left']

    def startState(self):
        #state=((x,y) of robot ,canvas state)
        return (self.pen,self.init)

    def isEnd(self,state):
        #norm state-end<=threshold
        canvas_state=np.copy(state[1])
        return np.linalg.norm(canvas_state-self.end)<=self.threshold

    def isBound(self,curr_pos):
        #pen x,y <=N
        return (curr_pos[0]<=self.N and curr_pos[1]<=self.N)

    def newposition(self, state,action):
        canvas_state=np.copy(state[1])
        pen=state[0].copy()
        if action=='up': pen[0]-=1
        if action=='down': pen[0]+=1
        if action=='left':pen[1]-=1
        if action=='right':pen[1]+=1
        if self.isBound(pen):
            canvas_state[pen[0],pen[1]]=1
        return (pen, canvas_state)

    def getSuccStateAction(self,state):
        #current_state:action->next_state
        #action=white pixel vicinity, up:down:center:left
        result=[]
        for action in self.actions:
            res_tuple=(state,action,self.newposition(state, action))
            result.append(res_tuple)
        return result

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
#print((sa[1])[2])
ea=model.isEnd((sa[1])[2])
print("isend=",ea)
#for item in sa:
#    print(item[0])
#    print(item[1])
#    print(item[2])
#    print("-----------------------")
