#!/usr/bin/env python3

from lib import NN

print("For this test, the first and last layer size must be 3!")
nn = NN.NN(False)

print("Weights")
print(nn.weights)

nn.forward([1,1,1])
print("\nOUTPUT: ", nn.results[len(nn.results)-1])
print("COST: ", sum(nn.cost([1,1,1])))
nn.backward([1,1,1])

nn.forward([1,1,1])
nn.backward([1,1,1])
nn.forward([1,1,1])
nn.backward([1,1,1])
nn.forward([1,1,1])
nn.backward([1,1,1])
nn.forward([1,1,1])
nn.backward([1,1,1])


print("\nNew Weights")
print(nn.weights)
nn.forward([1,1,1])
print("\nOUTPUT: ", nn.results[len(nn.results)-1])
print("COST: ", sum(nn.cost([1,1,1])))
