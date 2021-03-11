## Variables:
#
# self.layers               "map" of NN
# self.activationType       selects activation function to use
# self.outputType           selects final activation function to use
# self.weights              matrices holding neuron weights
# self.biases               arrays holding neuron biases
# self.results              arrays holding neuron results at every level

# self.learingRate          integer holding neuron learning rate
# self.decayRate            integer holding neuron learning "momentum"
# self.lastWeightDelta      weight changes of last time (to use with decay)
# self.lastBiasDelta        bias changes of last time (to use with decay)

import numpy as np

class NN:

    #
    # Init Function for Neural Network.
    # var load: true to load from file, false to create new
    #
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
            menu['4'] = "0 to 100%"
            for entry in menu.keys():
                print (entry, menu[entry])
            selection=input("Select an output type: ")
            self.outputType = menu[selection]
            print("\nOutput type: ", self.outputType, "\n")

            # Initialize Weights and Biases randomly
            self.weights = []
            self.biases = []
            self.lastWeightDelta = []
            self.lastBiasDelta = []
            self.learningRate = 0.001
            self.decayRate = 0.5
            for i in range(len(self.layers)):
                if(i != 0):
                    self.weights.append(np.random.rand(self.layers[i], self.layers[i-1]))
                    self.biases.append(np.random.rand(self.layers[i]))
                    self.lastWeightDelta.append(np.zeros((self.layers[i], self.layers[i-1])))
                    self.lastBiasDelta.append(np.zeros(self.layers[i]))
        return

    #
    # Propagate Forward function
    # generate output based on input
    #
    def forward(self, nnInput):
        self.results = []
        self.results.append(np.array(nnInput))
        for i in range(len(self.layers)-1):
            self.results.append(np.array(self.activation(np.matmul(self.weights[i], self.results[i]) + self.biases[i])))
        return self.results[len(self.results)-1]


    #
    # Back Propagation function
    # calculates by how much each value should change
    #
    def backward(self, ideal):
        output = self.results[len(self.results)-1]
         
        # calculate dC/da (based on output type)
        dcda = []
        if(self.outputType == "0 to 100%"):
            dcda = -ideal/output
        else:
            dcda = 2*(output-ideal)
        
        for i in range(len(self.layers)-1, 0, -1): # count down
            # calculate da/dz (based on activation function)
            # currently just based on ReLU
            dadz = (self.results[i] > 0) # 1 for > 0, 0 for <= 0

            # calculate dz/dW (activation level of previous layer)
            # TODO add support for more activation levels
            dzdw = self.results[i-1]
            
            # calculate required change in values
            dcdz = dcda*dadz
            dcdw = np.zeros([len(dcdz), len(dzdw)])
            for j in range(len(dcdz)):
                for k in range(len(dzdw)):
                    dcdw[j][k] = dcdz[j]*dzdw[k]

            dcdb = dcda*dadz                            #dz/db is always 1

            dcda = np.matmul(dcdz, self.weights[i-1])   # is really dC/d(a-1), but simpler to reassign var
            
            # Update values
            self.lastWeightDelta[i-1] = (-self.learningRate * dcdw) + (self.decayRate * np.array(self.lastWeightDelta[i-1]))
            self.lastBiasDelta[i-1] = (-self.learningRate * dcdb) + (self.decayRate * np.array(self.lastBiasDelta[i-1]))
            self.weights[i-1] += np.array(self.lastWeightDelta[i-1])
            self.biases[i-1] += self.lastBiasDelta[i-1]


        return


    #
    # Activation function, will be chosen by user
    #   Temp: just ReLU for now
    #   TODO add more functions
    def activation(self, nnInput):
        #ReLU
        for i in range(len(nnInput)):
            if(nnInput[i] < 0):
                nnInput[i] = 0
        return nnInput

    
    #
    # Cost function: auto detects based on function
    #
    def cost(self, ideal):
        output = self.results[len(self.results)-1]
        nnCost = []
        #Cross entropy for softMax function
        if (self.outputType == "0 to 100%"):
            for i in range(len(output)):
                nnCost.append(-ideal[i]*np.log(output[i]))
            return nnCost
        #Square Error for general case
        else:
            for i in range(len(output)):
                nnCost.append(pow(output[i] - ideal[i], 2))
            return nnCost
