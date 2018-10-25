#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

train_dataset = tv.datasets.MNIST(root='./data',train=True, transform=tv.transforms.ToTensor(), download=True)
test_dataset = tv.datasets.MNIST(root='./data',train=False, transform=tv.transforms.ToTensor(),download = True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=100, shuffle=False)

class NeuralNet(nn.Module):
	"""A Neural Network with a hidden layer"""
	def __init__(self, input_size,hidden_size,output_size):
		super(NeuralNet, self).__init__()
		self.layer1 = nn.Linear(input_size, hidden_size)
		self.layer2 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		output = self.layer1(x)
		output = self.relu(output)
		output = self.layer2(output)
		return output

input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 5

learning_rate = 0.001

model = NeuralNet(input_size,hidden_size, output_size)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def show_batch(batch):
    im = tv.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images,labels) in enumerate(train_loader):
		images = images.reshape(-1,28*28)
		show_batch(images)
		out = model(images)
		loss = lossFunction(out,labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
	correct = 0
	total = 0
	for images,labels in test_loader:
		images = images.reshape(-1,28*28)
		out = model(images)
		_,predicted = torch.max(out.data,1)
		total += labels.size(0)
		correct += (predicted==labels).sum().item()
		print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
