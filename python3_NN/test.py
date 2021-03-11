#!/usr/bin/env python3

from lib import NN

print("For this test, the first and last layer size must be 3!")

nn = NN.NN(False)

desiredVal = [1,0,1]
print("Desired Value: ", desiredVal)

nn.forward([1,1,1])
print("\nOUTPUT: ", nn.results[len(nn.results)-1])
print("COST: ", sum(nn.cost([1,1,1])))

for num in range(5):
	for i in range(2):
		for j in range(2):
			for k in range(2):
				nn.forward([i, j, k])
				nn.backward(desiredVal)

nn.forward([1,1,1])
print("\nOUTPUT: ", nn.results[len(nn.results)-1])
print("COST: ", sum(nn.cost(desiredVal)))
