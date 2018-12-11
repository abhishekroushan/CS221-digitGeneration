import matplotlib.pyplot as plt
import numpy as np

#Given a set of actions, and a start point, trace out the path and display the matrix

def visualize(start, actions, N):
	#start=(x,y) start coord
	#actions=list of actions from start ['left', 'down', 'up', 'left'] etc
	#N= size of canvas- ex 28
	matrix=np.zeros((N,N))
	pointer=list(start)
	matrix[pointer[0], pointer[1]]=1
	#for each action update the matrix
	for action in actions:
		if action =='up':
			pointer[0]-=1
		if action == 'down':
			pointer[0]+=1
		if action == 'right':
			pointer[1]+=1
		if action =='left':
			pointer[1]-=1
		matrix[pointer[0], pointer[1]]=1
	print("matrix")
	print(matrix)
	plt.imshow(matrix)
	plt.pause(10) 


#start state and actions
#start_pen=(7,7)
#canvas_size=28
#list_actions=['down', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'down', 'down', 'down', 'right', 'right', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 'right', 'right', 'right', 'right']
#visualize(start_pen, list_actions, canvas_size)