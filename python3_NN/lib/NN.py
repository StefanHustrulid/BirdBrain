## NN init:
# 0: new NN
# 1: load NN

## Variables:
#
# self.layers
# self.activationType
# self.outputType
# self.weights
# self.biases

import numpy as np

class NN:
    def __init__(self, load):
        if(load == True):
            #Load the neural network from a file with predetermined structure
            print("Loading from file...")

        else:
            # Initialize Neural Net layers
            self.layers=[]
            while (True):
                layer = int(input("How many Neurons in layer? (0 if done): "))
                if (layer > 0):
                    self.layers.append(layer)
                    continue
                break
            print("\nNeural net layer structure: ", self.layers, "\n")

            # Set NN activation function
            menu = {}
            menu['1'] = "Linear"
            menu['2'] = "Sigmoid"
            menu['3'] = "Tanh"
            menu['4'] = "ReLU"
            menu['5'] = "Leaky ReLU"
            for entry in menu.keys():
                print (entry, menu[entry])
            selection=input("Select an activation function: ")
            self.activationType = menu[selection]
            print("\nActivation function: ", self.activationType, "\n")

            # Set output type
            menu = {}
            menu['1'] = "0 to 1"
            menu['2'] = "0 to inf"
            menu['3'] = "-inf to inf"
            for entry in menu.keys():
                print (entry, menu[entry])
            selection=input("Select an output type: ")
            self.outputType = menu[selection]
            print("\nOutput type: ", self.outputType, "\n")

            # Initialize Weights and Biases randomly
            self.weights = []
            self.biases = []
            for i in range(len(self.layers)):
                if(i != 0):
                    self.weights.append(np.random.rand(self.layers[i], self.layers[i-1]))
                self.biases.append(np.random.rand(self.layers[i]))

            print("Neuron weights")
            print(self.weights, "\n\n\n")

            print("Neuron biases")
            print(self.biases)
