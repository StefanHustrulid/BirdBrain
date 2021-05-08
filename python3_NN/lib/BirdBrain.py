## Variables:
# self.layers               "map" of NN
# self.activationType       selects activation function to use
# self.outputType           selects final activation function to use
# self.weights              matrices holding neuron weights
# self.biases               arrays holding neuron biases
# self.constants            matrices holding adjustable constants for activation functions (Depends on activationType)
# self.results              arrays holding neuron results at every level
# self.learningRate          integer holding neuron learning rate
# self.decayRate            integer holding neuron learning "momentum"
# self.lastWeightDelta      weight changes of last time (to use with decay)
# self.lastBiasDelta        bias changes of last time (to use with decay)
import numpy as np
import json
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, fileName=''):
        # isNew is boolean
        NeuralNetworkStructure = []
        NeuralNetwork = []
        if fileName != '':
            #fileName = input("Neural Network Structure File: ")
            #NNSFile = json.open(fileName)
            with open(fileName) as fp:
                NeuralNetworkStructure = json.load(fp)
        else:
            layerTypes = {}
            layerTypes['1'] = "Convolution 2D"
            layerTypes['2'] = "Max Pooling 2D"
            layerTypes['3'] = "Min Pooling 2D"
            layerTypes['4'] = "Mean Pooling 2D"
            layerTypes['5'] = "L2 Pooling 2D"
            layerTypes['6'] = "Flatten 2D"
            layerTypes['7'] = "Inverse Flatten 2D"
            layerTypes['8'] = "Activation"
            layerTypes['9'] = "Fully Connected"
            layerTypes['10'] = "0-# for layer type information... Not Yet Implemented"
            layerTypes['11'] = "Done"
            activationOptions = {}
            activationOptions['1'] = "Linear"
            activationOptions['2'] = "Sigmoid"
            activationOptions['3'] = "Tanh"
            activationOptions['4'] = "ReLU"
            activationOptions['5'] = "LReLU"
            activationOptions['6'] = "PReLU"
            activationOptions['7'] = "ELU"
            activationOptions['8'] = "SELU"
            activationOptions['9'] = "GELU"
            activationOptions['10'] = "Softsign"
            activationOptions['11'] = "Softplus"
            activationOptions['12'] = "0-# for activation function information... Not Yet Implemented"
            fullyConnectedOptions = {}
            fullyConnectedOptions['1'] = "Linear"
            fullyConnectedOptions['2'] = "Sigmoid"
            fullyConnectedOptions['3'] = "Tanh"
            fullyConnectedOptions['4'] = "ReLU"
            fullyConnectedOptions['5'] = "LReLU"
            fullyConnectedOptions['6'] = "PReLU"
            fullyConnectedOptions['7'] = "ELU"
            fullyConnectedOptions['8'] = "SELU"
            fullyConnectedOptions['9'] = "GELU"
            fullyConnectedOptions['10'] = "Softsign"
            fullyConnectedOptions['11'] = "Softplus"
            fullyConnectedOptions['12'] = "Softmax"
            fullyConnectedOptions['13'] = "0-# for activation function information... Not Yet Implemented"
            layer = 0
            while True:
                for entry in layerTypes.keys():
                    print(entry, layerTypes[entry])
                layerType = layerTypes[input("Select Layer " + str(layer) + " Type:")]
                print("\n")
                if layerType == "Convolution 2D":
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Max Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Min Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Mean Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "L2 Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Inverse Flatten 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Flatten 2D":
                            # If there is a Flatten 2D Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [oC, oH, oW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            # If there is a Fully Connected Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [oC, oH, oW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                    # If loop runs into a Convolution 2D layer before a Activation or Fully Connected layer, it will initialize the weights.
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Activation" or NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            kernelShape = NeuralNetworkStructure[layer - l]["Kernel Shape"]
                            outputShape = NeuralNetworkStructure[layer - l]["Output Shape"]
                            distribution = NeuralNetworkStructure[layer - l]["Initial Weight Distribution"]
                            fanIn = kernelShape[2] * kernelShape[3]
                            fanOut = outputShape[1] * outputShape[2]
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, "Linear", distribution).tolist()
                            print("Layer", layer - l, "Kernels Initialized")
                            break
                    if layer == 0:
                        iH = int(input("Convolution 2D Layer Input Height"))
                        iW = int(input("Convolution 2D Layer Input Width"))
                        iC = int(input("Convolution 2D Layer Number of Input Channels"))
                        inputShape = [iC, iH, iW]
                    else:
                        inputShape = NeuralNetworkStructure[layer - 1]["Output Shape"]
                    iC = inputShape[0]
                    kH = int(input("Convolution 2D Layer Kernel Height: "))
                    kW = int(input("Convolution 2D Layer Kernel Width: "))
                    oC = int(input("Convolution 2D Layer Number of Kernels"))
                    kernelShape = [oC, iC, kH, kW]
                    while True:
                        distribution = input("Initial Convolution 2D Layer Kernel Weight Distribution (Gaussian or Uniform): ")
                        if distribution == "Gaussian" or distribution == "Uniform":
                            break
                        print("Did Not Understand. Please Try Again.\n")
                    stride = int(input("Convolution 2D Layer Stride: "))
                    while True:
                        ifBufferInput = input("Buffer (Yes or No): ")
                        if ifBufferInput == "Yes":
                            ifBuffer = True
                            break
                        elif ifBufferInput == "No":
                            ifBuffer = False
                            break
                        print("Did Not Understand. Please Try Again.\n")
                    outputShape = ConvolutionLayer2D.calcOutputShape(inputShape, kernelShape, stride, ifBuffer)
                    biases = ParameterInitializer.initializeBiases(outputShape)
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Kernel Shape"] = kernelShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    NeuralNetworkStructure[layer]["Stride"] = stride
                    NeuralNetworkStructure[layer]["If Buffer"] = ifBuffer
                    NeuralNetworkStructure[layer]["Initial Weight Distribution"] = distribution
                    NeuralNetworkStructure[layer]["Biases"] = biases.tolist()
                    print("\nLayer", layer, "Data:\n")
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    print("\nKernels will be generated when the next activation function is defined.\n")
                    layer = layer + 1
                elif layerType == "Max Pooling 2D" or layerType == "Min Pooling 2D" or layerType == "Mean Pooling 2D" or layerType == "L2 Pooling 2D":
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Max Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Min Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Mean Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "L2 Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Inverse Flatten 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Flatten 2D":
                            # If there is a Flatten 2D Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [oC, oH, oW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            # If there is a Fully Connected Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [oC, oH, oW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                    if layer == 0:
                        iH = int(input(layerType + " Layer Input Height: "))
                        iW = int(input(layerType + " Layer Input Width: "))
                        iC = int(input(layerType + " Layer Number of Input Channels: "))
                        inputShape = [iC, iH, iW]
                    else:
                        inputShape = NeuralNetworkStructure[layer - 1]["Output Shape"]
                    kH = int(input(layerType + " Layer Kernel Height: "))
                    kW = int(input(layerType + " Layer Kernel Width: "))
                    kernelShape = [kH, kW]
                    stride = int(input(layerType + " Layer Stride: "))
                    outputShape = PoolingLayer2D.calcOutputShape(inputShape, kernelShape, stride)
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Kernel Shape"] = kernelShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    NeuralNetworkStructure[layer]["Stride"] = stride
                    print("\nLayer", layer, "Data:\n")
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    layer = layer + 1
                elif layerType == "Flatten 2D":
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Max Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Min Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Mean Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "L2 Pooling 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Inverse Flatten 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Flatten 2D":
                            # If there is a Flatten 2D Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [iC, iH, iW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            # If there is a Fully Connected Layer before, insert an Inverse Flatten 2D layer
                            NeuralNetworkStructure.append({"Layer Type": "Inverse Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            print("Inserted Inverse Flatten 2D Layer\n")
                            while True:
                                print("Input size must equal", NeuralNetworkStructure[layer]["Input Shape"][0])
                                oH = int(input(layerType + " Layer Input Height: "))
                                oW = int(input(layerType + " Layer Input Width: "))
                                oC = int(input(layerType + " Layer Number of Input Channels: "))
                                if oH * oW * oC == NeuralNetworkStructure[layer]["Input Shape"][0]:
                                    NeuralNetworkStructure[layer]["Output Shape"] = [iC, iH, iW]
                                    break
                                print("Please try again.")
                            layer = layer + 1
                            break
                    if layer == 0:
                        iH = int(input("Flatten 2D Layer Input Height: "))
                        iW = int(input("Flatten 2D Layer Input Width: "))
                        iC = int(input("Flatten 2D Layer Number of Input Channels: "))
                        inputShape = [iC, iH, iW]
                    else:
                        inputShape = NeuralNetworkStructure[layer - 1]["Output Shape"]
                    outputShape = [iC*iH*iC, 1]
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    print("\nLayer", layer, "Data:\n")
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    layer = layer + 1
                elif layerType == "Inverse Flatten 2D":
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Flatten 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Max Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Min Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Mean Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "L2 Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Inverse Flatten 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                    if layer == 0:
                        iH = int(input("Inverse Flatten 2D Layer Input Height: "))
                    else:
                        minReq = NeuralNetworkStructure[layer - 1]["Output Shape"][0]
                        while True:
                            print("Inverse Flatten 2D Layer Input Height must be >=", minReq)
                            iH = int(input("Inverse Flatten 2D Layer Input Height: "))
                            if iH >= minReq:
                                break
                            print("Please Try Again.")
                    inputShape = [iH, 1]
                    while True:
                        print("Output size must equal", inputShape[0])
                        oH = int(input("Inverse Flatten 2D Layer Output Height: "))
                        oW = int(input("Inverse Flatten 2D Layer Output Width: "))
                        oC = int(input("Inverse Flatten 2D Layer Number of Output Channels: "))
                        if oH*oW*oC == inputShape[0]:
                            outputShape = [oC, oH, oW]
                            break
                        print("Please try again.")
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    print("\nLayer", layer, "Data:\n")
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    layer = layer + 1
                elif layerType == "Activation":
                    if layer == 0:
                        print("Possible formats:")
                        print("1 (iC, iH, iW)")
                        print("2 (iH, 1)")
                        type = int(input("Select Input Shape:"))
                        if type == 1:
                            iH = int(input("Activation Layer Input Height: "))
                            iW = int(input("Activation Layer Input Width: "))
                            iC = int(input("Activation Layer Number of Input Channels: "))
                            inputShape =[iC, iH, iW]
                        else:
                            iH = int(input("Activation Layer Input Height: "))
                            inputShape = [iH, 1]
                        outputShape = inputShape
                    else:
                        inputShape = NeuralNetworkStructure[layer - 1]["Output Shape"]
                        outputShape = inputShape
                    print("Activation Functions:")
                    for entry in activationOptions.keys():
                        print(entry, activationOptions[entry])
                    activationFunction = activationOptions[input("Select Activation Function: ")]
                    constants = ParameterInitializer.initializeConstants(outputShape, activationFunction)
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    NeuralNetworkStructure[layer]["Activation Function"] = activationFunction
                    NeuralNetworkStructure[layer]["Constants"] = constants.tolist()
                    # If loop runs into a Convolution 2D layer before a Activation or Fully Connected layer, it will initialize the weights.
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Activation" or NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            kernelShape = NeuralNetworkStructure[layer - l]["Kernel Shape"]
                            outputShape = NeuralNetworkStructure[layer - l]["Output Shape"]
                            distribution = NeuralNetworkStructure[layer - l]["Initial Weight Distribution"]
                            fanIn = kernelShape[2] * kernelShape[3]
                            fanOut = outputShape[1] * outputShape[2]
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, activationFunction, distribution).tolist()
                            print("Layer", layer - l, "Kernels Initialized")
                            break
                    print("\nLayer", layer, "Data:\n")
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    layer = layer + 1
                elif layerType == "Fully Connected":
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Flatten 2D":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Max Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Min Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Mean Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "L2 Pooling 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Inverse Flatten 2D":
                            NeuralNetworkStructure.append({"Layer Type": "Flatten 2D"})
                            NeuralNetworkStructure[layer]["Input Shape"] = NeuralNetworkStructure[layer - 1][
                                "Output Shape"]
                            NeuralNetworkStructure[layer]["Output Shape"] = [
                                np.prod(NeuralNetworkStructure[layer]["Input Shape"]), 1]
                            print("Inserted Flatten 2D Layer\n")
                            layer = layer + 1
                            break
                    if layer == 0:
                        iN = int(input("Fully Connected Layer Number of Input Neurons: "))
                    else:
                        minReq = NeuralNetworkStructure[layer - 1]["Output Shape"][0]
                        while True:
                            print("Number of Input Neurons must be >=", minReq)
                            iN = int(input("Fully Connected Layer Number of Input Neurons: "))
                            if iN >= minReq:
                                break
                            print("Please Try Again.")
                    inputShape = [iN, 1]
                    oN = int(input("Fully Connected Layer Number of Output Neurons: "))
                    for entry in fullyConnectedOptions.keys():
                        print(entry, fullyConnectedOptions[entry])
                    activationFunction = fullyConnectedOptions[input("Select Activation Function: ")]
                    # If loop runs into a Convolution 2D layer before a Activation or Fully Connected layer, it will initialize the weights.
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Activation" or NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            kernelShape = NeuralNetworkStructure[layer - l]["Kernel Shape"]
                            outputShape = NeuralNetworkStructure[layer - l]["Output Shape"]
                            distribution = NeuralNetworkStructure[layer - l]["Initial Weight Distribution"]
                            fanIn = kernelShape[2] * kernelShape[3]
                            fanOut = outputShape[1] * outputShape[2]
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, activationFunction, distribution).tolist()
                            print("Layer", layer - l, "Kernels Initialized")
                            break
                    outputShape = [oN, 1]
                    while True:
                        distribution = input("Initial Fully Connected Layer Weight Distribution (Gaussian or Uniform): ")
                        if distribution == "Gaussian" or distribution == "Uniform":
                            break
                        print("Did Not Understand. Please Try Again.\n")
                    weights = ParameterInitializer.initializeWeights(iN, oN, [oN, iN], activationFunction, distribution)
                    biases = ParameterInitializer.initializeBiases(outputShape)
                    constants = ParameterInitializer.initializeConstants(outputShape, activationFunction)
                    NeuralNetworkStructure.append({"Layer Type": layerType})
                    NeuralNetworkStructure[layer]["Input Shape"] = inputShape
                    NeuralNetworkStructure[layer]["Output Shape"] = outputShape
                    NeuralNetworkStructure[layer]["Activation Function"] = activationFunction
                    NeuralNetworkStructure[layer]["Weights"] = weights.tolist()
                    NeuralNetworkStructure[layer]["Biases"] = biases.tolist()
                    NeuralNetworkStructure[layer]["Constants"] = constants.tolist()
                    for entry in NeuralNetworkStructure[layer].keys():
                        print(entry, ":", NeuralNetworkStructure[layer][entry])
                    layer = layer + 1
                elif layerType == "Done":
                    # If loop runs into a Convolution 2D layer before a Activation or Fully Connected layer, it will initialize the weights.
                    for l in range(1, layer + 1):
                        if NeuralNetworkStructure[layer - l]["Layer Type"] == "Activation" or \
                                NeuralNetworkStructure[layer - l]["Layer Type"] == "Fully Connected":
                            break
                        elif NeuralNetworkStructure[layer - l]["Layer Type"] == "Convolution 2D":
                            kernelShape = NeuralNetworkStructure[layer - l]["Kernel Shape"]
                            outputShape = NeuralNetworkStructure[layer - l]["Output Shape"]
                            distribution = NeuralNetworkStructure[layer - l]["Initial Weight Distribution"]
                            fanIn = kernelShape[2] * kernelShape[3]
                            fanOut = outputShape[1] * outputShape[2]
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, "Linear", distribution).tolist()
                            print("Layer", layer - l, "Kernels Initialized")
                            break
                    break
        for l in np.arange(0, len(NeuralNetworkStructure)):
            print("\nLayer", l, "Information:")
            for entry in NeuralNetworkStructure[l].keys():
                print(entry, ":", NeuralNetworkStructure[l][entry])
        # Use NeuralNetworkStructure to build NeuralNetwork
        for layer in NeuralNetworkStructure:
            if layer["Layer Type"] == "Convolution 2D":
                inputShape = layer["Input Shape"]
                kernels = layer["Kernels"]
                biases = layer["Biases"]
                stride = layer["Stride"]
                ifBuffer = layer["If Buffer"]
                NeuralNetwork.append(ConvolutionLayer2D(inputShape, kernels, biases, stride, ifBuffer))
            elif layer["Layer Type"] == "Max Pooling 2D":
                inputShape = layer["Input Shape"]
                kernelShape = layer["Kernel Shape"]
                stride = layer["Stride"]
                NeuralNetwork.append(MaxPooling2D(inputShape, kernelShape, stride))
            elif layer["Layer Type"] == "Min Pooling 2D":
                inputShape = layer["Input Shape"]
                kernelShape = layer["Kernel Shape"]
                stride = layer["Stride"]
                NeuralNetwork.append(MinPooling2D(inputShape, kernelShape, stride))
            elif layer["Layer Type"] == "Mean Pooling 2D":
                inputShape = layer["Input Shape"]
                kernelShape = layer["Kernel Shape"]
                stride = layer["Stride"]
                NeuralNetwork.append(MeanPooling2D(inputShape, kernelShape, stride))
            elif layer["Layer Type"] == "L2 Pooling 2D":
                inputShape = layer["Input Shape"]
                kernelShape = layer["Kernel Shape"]
                stride = layer["Stride"]
                NeuralNetwork.append(L2Pooling2D(inputShape, kernelShape, stride))
            elif layer["Layer Type"] == "Flatten 2D":
                inputShape = layer["Input Shape"]
                NeuralNetwork.append(FlattenLayer2D(inputShape))
            elif layer["Layer Type"] == "Inverse Flatten 2D":
                outputShape = layer["Output Shape"]
                NeuralNetwork.append(InverseFlattenLayer2D(outputShape))
            elif layer["Layer Type"] == "Activation":
                constants = layer["Constants"]
                activationFunction = layer["Activation Function"]
                NeuralNetwork.append(ActivationLayer(constants, activationFunction))
            elif layer["Layer Type"] == "Fully Connected":
                weights = layer["Weights"]
                biases = layer["Biases"]
                constants = layer["Constants"]
                activationFunction = layer["Activation Function"]
                NeuralNetwork.append(FullyConnectedLayer(weights, biases, constants, activationFunction))
        self.NeuralNetworkStructure = NeuralNetworkStructure
        self.NeuralNetwork = NeuralNetwork

    def exportNeuralNetwork(self):
        # Saves the NeuralNetworkStructure as a json file
        Name = input("Neural Network Structure File Name: ")
        with open(Name, 'w') as fp:
            json.dump(self.NeuralNetworkStructure, fp, indent=4)

    def runNeuralNetwork(self, inputActivations):
        # inputActivations is AL lists of inputShape
        # AL is number of input layers / number of layers where inputs are added
        # inputShape is (N, c, h, w) or (N, h, 1)
        # after the first layer inputShape must be (N, h, 1)
        # activations is shape (L, N, activationShape)
        inputSet = 0
        activations = [np.array(inputActivations[inputSet])]
        for layer in np.arange(0, np.size(self.NeuralNetwork)):
            if layer > 0:
                if self.NeuralNetworkStructure[layer]["Layer Type"] == "Fully Connected" or self.NeuralNetworkStructure[layer]["Layer Type"] == "Inverse Flatten 2D":
                    if self.NeuralNetworkStructure[layer]["Input Shape"][0] > self.NeuralNetworkStructure[layer - 1]["Output Shape"][0]:
                        inputSet = inputSet + 1
                        activations[layer] = np.append(activations[layer], inputActivations[inputSet], 1)
            activations.append(self.NeuralNetwork[layer].forward(activations[layer]))
        return activations

    def costFromOutput(self, actualOutput, idealOutput):
        # actualOutput and idealOutput can be any shape as long as N is the 0 dim
        costFunction = "Squared Error"
        for layer in np.arange(1, np.size(self.NeuralNetwork)):
            if self.NeuralNetworkStructure[-layer]["Layer Type"] == "Convolution 2D":
                break
            elif self.NeuralNetworkStructure[-layer]["Layer Type"] == "Activation":
                break
            elif self.NeuralNetworkStructure[-layer]["Layer Type"] == "Fully Connected":
                if self.NeuralNetworkStructure[-layer]["Activation Function"] == "Softmax":
                    costFunction = "Cross Entropy"
                    break
                else:
                    break
        if costFunction == "Squared Error":
            return self.squaredError(actualOutput, idealOutput)
        else:
            return self.crossEntropy(actualOutput, idealOutput)

    def costFromInput(self, inputActivations, idealOutput):
        activations = self.runNeuralNetwork(inputActivations)
        actualOutput = activations[-1]
        return self.costFromOutput(actualOutput, idealOutput)

    def dCostdOutputsFromOutput(self, actualOutput, idealOutput):
        # 1st determine the cost function
        costFunction = "Squared Error"
        for layer in np.arange(1, np.size(self.NeuralNetwork) + 1):
            if self.NeuralNetworkStructure[-layer]["Layer Type"] == "Convolution 2D":
                break
            elif self.NeuralNetworkStructure[-layer]["Layer Type"] == "Activation":
                break
            elif self.NeuralNetworkStructure[-layer]["Layer Type"] == "Fully Connected":
                if self.NeuralNetworkStructure[-layer]["Activation Function"] == "Softmax":
                    costFunction = "Cross Entropy"
                    break
                else:
                    break
        # calculate the basic dCda
        if costFunction == "Cross Entropy":
            cDerivativeO = self.crossEntropyDerivative(actualOutput, idealOutput)
        else:
            cDerivativeO = self.squaredErrorDerivative(actualOutput, idealOutput)
        # adjust format type if needed
        # if output is flat shape then should tile so that cDerivativeO is N square matrices
        for layer in np.arange(1, np.size(self.NeuralNetwork) + 1):
            if self.NeuralNetworkStructure[-layer]["Layer Type"] == "Fully Connected":
                cDerivativeO = np.tile(cDerivativeO, (1, 1, cDerivativeO.shape[1]))
                break
            if self.NeuralNetworkStructure[-layer]["Layer Type"] == "Flatten 2D":
                cDerivativeO = np.tile(cDerivativeO, (1, 1, cDerivativeO.shape[1]))
                break
            if self.NeuralNetworkStructure[-layer]["Layer Type"] != "Activation":
                break
        return cDerivativeO

    def dCostdOutputsFromInput(self, inputActivations, idealOutput):
        activations = self.runNeuralNetwork(inputActivations)
        actualOutput = activations[-1]
        return self.dCostdOutputsFromOutput(actualOutput, idealOutput)

    def calcGradient(self, activations, cDerivativeO):
        # does not calc the first dCostdOutputs
        # calculates gradients/changes for every layer
        # modify for added inputs
        gradient = []
        cDerivativeA = cDerivativeO
        for layer in np.arange(1, len(self.NeuralNetwork) + 1):
            gradient.insert(0, self.NeuralNetwork[-layer].backprop(activations[-layer - 1], cDerivativeA))
            cDerivativeA = gradient[0][0]
            if layer == len(self.NeuralNetwork):
                return gradient
            # If there is a layer of added inputs, delete the extra cDerivativeA
            if self.NeuralNetworkStructure[-layer]["Layer Type"] == "Fully Connected" or self.NeuralNetworkStructure[-layer]["Layer Type"] == "Inverse Flatten 2D":
                difference = self.NeuralNetworkStructure[-layer]["Input Shape"][0] - self.NeuralNetworkStructure[-layer - 1]["Output Shape"][0]
                if difference > 0:
                    cDerivativeA = np.delete(cDerivativeA, np.arange(self.NeuralNetworkStructure[-layer]["Input Shape"][0] - difference, self.NeuralNetworkStructure[-layer]["Input Shape"][0]), axis=1)
                    cDerivativeA = np.delete(cDerivativeA, np.arange(self.NeuralNetworkStructure[-layer]["Input Shape"][0] - difference, self.NeuralNetworkStructure[-layer]["Input Shape"][0]), axis=2)
        return gradient


    def step(self, gradient, learningRate):
        # will adjust the NN and NNstructure for a step
        for layer in np.arange(0, len(self.NeuralNetwork)):
            if self.NeuralNetworkStructure[layer]["Layer Type"] == "Convolution 2D":
                dCdW = gradient[layer][1]
                dCdb = gradient[layer][2]
                self.NeuralNetworkStructure[layer]["Kernels"] = self.NeuralNetworkStructure[layer]["Kernels"] - (learningRate*dCdW)
                self.NeuralNetworkStructure[layer]["Biases"] = self.NeuralNetworkStructure[layer]["Biases"] - (learningRate*dCdb)
                self.NeuralNetwork[layer].kernels = self.NeuralNetwork[layer].kernels - (learningRate * dCdW)
                self.NeuralNetwork[layer].biases = self.NeuralNetwork[layer].biases - (learningRate * dCdb)
                self.NeuralNetwork[layer].createConvolutionMatrix() # causes the convolution layer to update it's convolution matrix
            elif self.NeuralNetworkStructure[layer]["Layer Type"] == "Activation":
                dCdk = gradient[layer][1]
                self.NeuralNetworkStructure[layer]["Constants"] = self.NeuralNetworkStructure[layer]["Constants"] - (learningRate * dCdk)
                self.NeuralNetwork[layer].constants = self.NeuralNetwork[layer].constants - (learningRate * dCdk)

            elif self.NeuralNetworkStructure[layer]["Layer Type"] == "Fully Connected":
                dCdW = gradient[layer][1]
                dCdb = gradient[layer][2]
                dCdk = gradient[layer][3]
                self.NeuralNetworkStructure[layer]["Weights"] = self.NeuralNetworkStructure[layer]["Weights"] - (learningRate * dCdW)
                self.NeuralNetworkStructure[layer]["Biases"] = self.NeuralNetworkStructure[layer]["Biases"] - (learningRate * dCdb)
                self.NeuralNetworkStructure[layer]["Constants"] = self.NeuralNetworkStructure[layer]["Constants"] - (learningRate * dCdk)
                self.NeuralNetwork[layer].weights = self.NeuralNetwork[layer].weights - (learningRate * dCdW)
                self.NeuralNetwork[layer].biases = self.NeuralNetwork[layer].biases - (learningRate * dCdb)
                self.NeuralNetwork[layer].constants = self.NeuralNetwork[layer].constants - (learningRate * dCdk)

    def gradientStep(self, activations, cDerivativeO, learningRate):
        gradient = self.calcGradient(activations, cDerivativeO)
        self.step(gradient, learningRate)
        # Returns the average change for plotting purposes

    def extractTrainingData(self, testData, percent):
        # testData is [inputs, idealOutputs]
        # inputs is [(N, inputShape), (N, addedH, 1), (N, addedH, 1),...]
        # inputShape can be shape (c, h, w) or (h, 1)
        # idealOutputs is (N, outputShape)
        # outputShape can be shape (c, h, w) or (h, 1)
        # percent is a float > 0 & <= 100
        # will extract n% of Test data to use as training data
        # shuffle the testData
        inputs = testData[0]
        idealOutputs = testData[1]
        shuffledInputs = []
        for layer in inputs:
            np.random.shuffle(layer)
            shuffledInputs.append(layer)
        shuffledIdealOutputs = idealOutputs
        np.random.shuffle(shuffledIdealOutputs)
        # determine the trainData size
        trainingSize = int(np.floor(idealOutputs.shape[0]*percent/100))
        if trainingSize == 0:
            trainingSize = 1
        # extract trainData from shuffled testData
        trainingInputs = []
        for layer in shuffledInputs:
            trainingInputs.append(layer[0:trainingSize])
        trainingIdealOutputs = shuffledIdealOutputs[0:trainingSize]
        trainingData = [trainingInputs, trainingIdealOutputs]
        return trainingData


    def generateBatches(self, trainingData, batchSize):
        # trainingData is [inputs, idealOutputs]
        # inputs is [(N, inputShape), (N, addedH, 1), (N, addedH, 1),...]
        # inputShape can be shape (c, h, w) or (h, 1)
        # idealOutputs is (N, outputShape)
        # outputShape can be shape (c, h, w) or (h, 1)
        # will split training data into batches
        # batch is [batchesOfInputs, batchesOfIdealOutputs]
        # batchesOfInputs is array of mulitple [(batchSize, inputShape), (batchSize, addedH, 1), (batchSize, addedH, 1),...]
        # batchesOfIdealOutputs is array of multiple (batchSize, outputshape)
        inputs = trainingData[0]
        idealOutputs = trainingData[1]
        N = idealOutputs.shape[0]
        shuffledInputs = []
        for layer in inputs:
            np.random.shuffle(layer)
            shuffledInputs.append(layer)
        shuffledIdealOutputs = idealOutputs
        np.random.shuffle(shuffledIdealOutputs)
        numberOfBatches = int(np.floor(N / batchSize))
        lastBatchSize = np.remainder(N, batchSize)
        batch = []
        for b in np.arange(0, numberOfBatches):
            b = int(b)
            batchInputs = []
            for layer in shuffledInputs:
                batchInputs.append(layer[batchSize * b:batchSize * (b + 1)])
            batch.append([batchInputs, shuffledIdealOutputs[batchSize * b:batchSize * (b + 1)]])
        if lastBatchSize > 0:
            b = numberOfBatches
            batchInputs = []
            for layer in shuffledInputs:
                batchInputs.append(layer[batchSize*b:])
            batch.append([batchInputs, shuffledIdealOutputs[batchSize*b:]])
        return batch

    def gradientDescentNeuralNetwork(self, testData, learningRate, acceptableCost, maxNumberOfSteps, percentForTraining, batchSize, plot=False):
        # will train the NN with gradient descent
        trainingData = self.extractTrainingData(testData, percentForTraining)
        batch = self.generateBatches(trainingData, batchSize)
        step = 0
        totalCost = []
        totalCost = np.append(totalCost, np.mean(self.costFromInput(testData[0], testData[1])[1]))
        while step <= maxNumberOfSteps and totalCost[step] >= acceptableCost:
            for dataSet in batch:
                # costFromInput returns [cost, totalCost]
                # where totalCost is a N-long array
                # so average totalCost to get a single number for that batch
                # then append to the list of totalCost for each step.
                if totalCost[step] <= acceptableCost:
                    break
                if step >= maxNumberOfSteps:
                   break
                activations = self.runNeuralNetwork(dataSet[0])
                cDerivativeO = self.dCostdOutputsFromOutput(activations[-1], dataSet[1])
                self.gradientStep(activations, cDerivativeO, learningRate)
                totalCost = np.append(totalCost, np.mean(self.costFromInput(testData[0], testData[1])[1]))
                step = step + 1
            batch = self.generateBatches(trainingData, batchSize)
        if plot == True:
            plt.plot(totalCost)

    @staticmethod
    def crossEntropy(actualOutput, idealOutput):
        # inputs are shape (N, shape)
        # cost is shape (N, shape)
        # total Cost is shape(1, N)
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        # cost is element wise
        cost = -idealOutput * np.log(actualOutput)
        # totalCost is shape (1, N)
        totalCost = []
        for n in np.arange(0, cost.shape[0]):
            totalCost = np.append(totalCost, sum(cost[n]))
        return cost, totalCost

    @classmethod
    def crossEntropyDerivative(cls, actualOutput, idealOutput):
        # actualOutput is shape (n, Shape)
        # idealOutput is same shape as actualOutput
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cDerivativeA = -1/actualOutput
        #cDerivativeA = cls.diagMat(-idealOutput / actualOutput)
        return cDerivativeA

    @staticmethod
    def squaredError(actualOutput, idealOutput):
        # inputs are shape (N, shape)
        # cost is shape (N, shape)
        # total Cost is shape(1, N)
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cost = np.power(actualOutput - idealOutput, 2)
        totalCost = []
        for n in np.arange(0, cost.shape[0]):
            totalCost = np.append(totalCost, sum(cost[n]))
        return cost, totalCost

    @classmethod
    def squaredErrorDerivative(cls, actualOutput, idealOutput):
        # actualOutput is shape (n, Shape)
        # idealOutput is same shape as actualOutput
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cDerivativeA = 2*(actualOutput - idealOutput)
        #cDerivativeA = cls.diagMat(2 * (actualOutput - idealOutput))
        return cDerivativeA


class ParameterInitializer:

    @classmethod
    def initializeConstants(cls, outputShape, activationFunction):
        # outputShape is either (fanOut, 1) or (oC, oH, oW)
        # activationFunction is a string
        if activationFunction == "LReLU":
            print("Note: The value chosen will be uniform for this layer.")
            print("Please choose a number >0 and <1")
            constant = float(input("The constant that will be used for the LReLU activation function of this layer:"))
            return constant*np.ones(outputShape)
        elif activationFunction == "ELU":
            print("Note: The value chosen will be uniform for this layer.")
            print("Please choose a number >0")
            constant = float(input("The constant that will be used for the ELU activation function of this layer:"))
            return constant*np.ones(outputShape)
        elif activationFunction == "PReLU":
            # It should be fine that they all start the same as randomizing the weights will eliminate redundant symmetry
            print("Note: The value chosen will be uniform for this layer.")
            print("Please choose a number >0 and <1")
            constant = float(input("The initial constant that will be used for the PReLU activation function of this layer:"))
            return constant * np.ones(outputShape)
        else:
            return np.zeros(outputShape)

    @classmethod
    def initializeBiases(cls, outputShape):
        # outputShape is either (fanOut, 1) or (oC, oH,oW)
        return np.zeros(outputShape)

    @classmethod
    def initializeWeights(cls, fanIn, fanOut, weightShape, activationFunction, distribution):
        # if fully connected weightShape is (fanOut, fanIn)
        # else weightShape is (oC, iC, kW, kW), fanIn is kH*kW, and fanOut is oH*oW
        # activation Function is a string
        # distribution is either "Gaussian" or "Uniform"
        if activationFunction == "ELU" or activationFunction == "SELU" or activationFunction == "GELU":
            if distribution == "Gaussian":
                return cls.LeCunGaussian(fanIn, weightShape)
            elif distribution == "Uniform":
                return cls.LeCunGaussian(fanIn, weightShape)
        elif activationFunction == "ReLU" or activationFunction == "LReLU" or activationFunction == "PReLU":
            if distribution == "Gaussian":
                return cls.HeGaussian(fanIn, weightShape)
            elif distribution == "Uniform":
                return cls.HeGaussian(fanIn, weightShape)
        else:
            if distribution == "Gaussian":
                return cls.XavierGaussian(fanIn, fanOut, weightShape)
            elif distribution == "Uniform":
                return cls.XavierGaussian(fanIn, fanOut, weightShape)

    @staticmethod
    def XavierGaussian(fanIn, fanOut, weightShape):
        # fanIn is an int
        # fanOut is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        stdDev = np.sqrt(2/(fanIn + fanOut))
        mean = 0
        weights = np.random.normal(mean, stdDev, weightShape)
        return weights

    @staticmethod
    def XavierUniform(fanIn, fanOut, weightShape):
        # fanIn is an int
        # fanOut is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        var = np.sqrt(2*6/(fanIn + fanOut))
        weights = np.random.uniform(-var/2, var/2, weightShape)
        return weights

    @staticmethod
    def HeGaussian(fanIn, weightShape):
        # fanIn is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        stdDev = np.sqrt(2/fanIn)
        mean = 0
        weights = np.random.normal(mean, stdDev, weightShape)
        return weights

    @staticmethod
    def HeUniform(fanIn, weightShape):
        # fanIn is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        var = np.sqrt(2*6/fanIn)
        weights = np.random.uniform(-var/2, var/2, weightShape)
        return weights

    @staticmethod
    def LeCunGaussian(fanIn, weightShape):
        # fanIn is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        stdDev = np.sqrt(1/fanIn)
        mean = 0
        weights = np.random.normal(mean, stdDev, weightShape)
        return weights

    @staticmethod
    def LeCunUniform(fanIn, weightShape):
        # fanIn is an int
        # weightShape is either (fanOut, fanIn) or (oC, iC, kH, kW)
        var = np.sqrt(3/fanIn)
        weights = np.random.uniform(-var/2, var/2, weightShape)
        return weights


class ActivationLayer:

    def __init__(self, constants, activationFunction):
        self.constants = constants
        self.activationFunction = activationFunction

    def forward(self, z):
        # z is any shape
        # output is same shape as z
        if self.activationFunction == "linear":
            return self.linearActivation(z)
        elif self.activationFunction == "sigmoid":
            return self.sigmoidActivation(z)
        elif self.activationFunction == "tanh":
            return self.tanhActivation(z)
        elif self.activationFunction == "ReLU":
            return self.ReLUActivation(z)
        elif self.activationFunction == "LReLU":
            return self.LReLUActivation(z, self.constants)
        elif self.activationFunction == "PReLU":
            return self.PReLUActivation(z, self.constants)
        elif self.activationFunction == "ELU":
            return self.ELUActivation(z, self.constants)
        elif self.activationFunction == "SELU":
            return self.SELUActivation(z)
        elif self.activationFunction == "GELU":
            return self.GELUActivtion(z)
        elif self.activationFunction == "softsign":
            return self.softsignActivation(z)
        elif self.activationFunction == "softplus":
            return self.softplusActivation(z)
        elif self.activationFunction == "softmax":
            return self.softmaxActivation(z)

    def backprop(self, z, cDerivativeA):
        cDerivativePrevZ = cDerivativeA*self.activationDerivativeZ(z)
        cDerivativeK = cDerivativeA*self.activationDerivativeK(z)
        return cDerivativePrevZ, cDerivativeK

    def activationDerivativeZ(self, z):
        if self.activationFunction == "linear":
            return self.linearDerivative(z)
        elif self.activationFunction == "sigmoid":
            return self.sigmoidDerivative(z)
        elif self.activationFunction == "tanh":
            return self.tanhDerivative(z)
        elif self.activationFunction == "ReLU":
            return self.ReLUDerivative(z)
        elif self.activationFunction == "LReLU":
            return self.LReLUDerivative(z, self.constants)
        elif self.activationFunction == "PReLU":
            return self.PReLUDerivative(z, self.constants)[0]
        elif self.activationFunction == "ELU":
            return self.ELUDerivative(z, self.constants)
        elif self.activationFunction == "SELU":
            return self.SELUDerivative(z)
        elif self.activationFunction == "GELU":
            return self.GELUDerivative(z)
        elif self.activationFunction == "softsign":
            return self.softsignDerivative(z)
        elif self.activationFunction == "softplus":
            return self.softplusDerivative(z)

    def activationDerivativeK(self, z):
        if self.activationFunction == "PReLU":
            return self.PReLUDerivative(z, self.constants)[1]
        else:
            return np.zeros(self.constants.shape)

    @staticmethod
    def linearActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = z
        return activation

    @staticmethod
    def linearDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        aDerivativeZ = np.ones(z.shape)
        return aDerivativeZ

    @staticmethod
    def sigmoidActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = 1 / (1 + np.exp(-z))
        return activation

    @classmethod
    def sigmoidDerivative(cls, z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = cls.sigmoidActivation(z)
        aDerivativeZ = activation * (1 - activation)
        return aDerivativeZ

    @staticmethod
    def tanhActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = np.tanh(z)
        return activation

    @staticmethod
    def tanhDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        aDerivativeZ = 1 - np.power(z, 2)
        return aDerivativeZ

    @staticmethod
    def ReLUActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = z * (z >= 0)
        return activation

    @staticmethod
    def ReLUDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        aDerivativeZ = 1 * (z >= 0)
        return aDerivativeZ

    @staticmethod
    def LReLUActivation(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats from 0-1 same shape as z
        z = np.array(z)
        activation = (z * (z >= 0)) + (constant * z * (z < 0))
        return activation

    @staticmethod
    def LReLUDerivative(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats from 0-1 same shape as z
        z = np.array(z)
        aDerivativeZ = (1 * (z >= 0)) + (constant * (z < 0))
        return aDerivativeZ

    @staticmethod
    def PReLUActivation(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats from 0-1 same shape as z
        z = np.array(z)
        constant = np.array(constant)
        activation = (z * (z >= 0)) + (constant * z * (z < 0))
        return activation

    @staticmethod
    def PReLUDerivative(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats from 0-1 same shape as z
        z = np.array(z)
        constant = np.array(constant)
        aDerivativeZ = (1 * (z >= 0)) + (constant * (z < 0))
        aDerivativeK = z * (z < 0)
        return aDerivativeZ, aDerivativeK

    @staticmethod
    def ELUActivation(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats >0 same shape as z
        z = np.array(z)
        activation = (z * (z >= 0)) + (constant * (np.exp(z) - 1) * (z < 0))
        return activation

    @staticmethod
    def ELUDerivative(z, constant):
        # z is any shape
        # output is same shape as z
        # constant is an array of floats >0 same shape as z
        z = np.array(z)
        aDerivativeZ = (1 * (z >= 0)) + (constant * np.exp(z) * (z < 0))
        return aDerivativeZ

    @staticmethod
    def SELUActivation(z):
        # z is any shape
        # output is same shape as z
        alphaConstant = 1.6732632423543772848170429916717
        lambdaConstant = 1.0507009873554804934193349852946
        z = np.array(z)
        activation = lambdaConstant * z * ((z >= 0) + (alphaConstant * (np.exp(z) - 1) * (z < 0)))
        return activation

    @staticmethod
    def SELUDerivative(z):
        # z is any shape
        # output is same shape as z
        alphaConstant = 1.6732632423543772848170429916717
        lambdaConstant = 1.0507009873554804934193349852946
        z = np.array(z)
        aDerivativeZ = lambdaConstant * ((z >= 0) + (alphaConstant * np.exp(z) * (z < 0)))
        return aDerivativeZ

    @staticmethod
    def GELUActivtion(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + (0.044715 * np.power(z, 3)))))
        return activation

    @staticmethod
    def GELUDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        aDerivativeZ = (0.5 * np.tanh((0.0356774 * np.power(z, 3)) + (0.797885 * z))) + (((0.535161 * np.power(z, 3)) + (0.398942 * z)) * np.power(np.sech((0.0356774 * np.power(z, 3)) + (0.797885 * z)), 2)) + 0.5
        return aDerivativeZ

    @staticmethod
    def softsignActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = z / (np.abs(z) + 1)
        return activation

    @staticmethod
    def softsignDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        absolute = np.abs(z)
        aDerivativeZ = ((z >= 0) * (absolute + 1 - z) / np.power(np.abs(z) + 1, 2)) + ((z < 0) * (np.abs(z) + 1 + z) / np.power(absolute + 1, 2))
        return aDerivativeZ

    @staticmethod
    def softplusActivation(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = np.log(1 + np.exp(z))
        return activation

    @classmethod
    def softplusDerivative(cls,z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        aDerivativeZ = cls.sigmoidActivation(z)
        return aDerivativeZ

    @staticmethod
    def softmaxActivation(z):
        # z is shape (N, iH, 1)
        # output is shape (N, iH, 1)
        z = np.array(z)
        activation = z / np.sum(z, 0)
        return activation

    @classmethod
    def softmaxDerivative(cls, z):
        # z is shape (N, iH, 1)
        # output is shape (N, iH, 1)
        z = np.array(z)
        activation = cls.softmaxActivation(z)
        diagMatrix = cls.diagMat(activation)
        softmaxMatrix = np.tile(activation, (1, 1, 2))
        aDerivativeZ = diagMatrix - np.matmul(softmaxMatrix, np.transpose(softmaxMatrix, (0, 2, 1)))
        return aDerivativeZ

    @staticmethod
    def diagMat(inputStack):
        # Custom method that takes (N, m, 1) or (N, 1, m) array
        # And outputs (N, m, m) diagonal matrices
        inputStack = np.array(inputStack)
        diagMatrix = []
        for N in inputStack:
            nDiagMatrix = np.diagflat(N)
            diagMatrix.append(nDiagMatrix)
        diagMatrix = np.array(diagMatrix)
        return diagMatrix

    @staticmethod
    def invDiagMat(diagMatStack):
        # Custom method that takes diagonal matrices with shape (N, m, m)
        # And outputs array with shape (N, m, 1)
        diagMatStack = np.array(diagMatStack)
        outputArray = np.sum(diagMatStack, 2, keepdims=True)
        return outputArray



class FullyConnectedLayer(ActivationLayer):

    def __init__(self, weights, biases, constants, activationFunction):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.constants = np.array(constants)
        self.activationFunction = activationFunction

    def forward(self, inputActivations):
        # inputActivations is shape (N, iH, 1)
        # output is shape (N, oH, 1)
        z = self.calcZ(self.weights, self.biases, inputActivations)
        if self.activationFunction == "linear":
            return self.linearActivation(z)
        elif self.activationFunction == "sigmoid":
            return self.sigmoidActivation(z)
        elif self.activationFunction == "tanh":
            return self.tanhActivation(z)
        elif self.activationFunction == "ReLU":
            return self.ReLUActivation(z)
        elif self.activationFunction == "LReLU":
            return self.LReLUActivation(z, self.constants)
        elif self.activationFunction == "PReLU":
            return self.PReLUActivation(z, self.constants)
        elif self.activationFunction == "ELU":
            return self.ELUActivation(z, self.constants)
        elif self.activationFunction == "SELU":
            return self.SELUActivation(z)
        elif self.activationFunction == "GELU":
            return self.GELUActivtion(z)
        elif self.activationFunction == "softsign":
            return self.softsignActivation(z)
        elif self.activationFunction == "softplus":
            return self.softplusActivation(z)
        elif self.activationFunction == "softmax":
            return self.softmaxActivation(z)

    @staticmethod
    def calcZ(weights, biases, prevActivations):
        # weights is shape (n, m)
        # biases is shape (n, 1)
        # prevActivations is shape (N, m, 1)
        # calculates z = (W x a) + b
        # z will be shape (N, n, 1)
        weights = np.array(weights)
        prevActivations = np.array(prevActivations)
        biases = np.array(biases)
        z = np.transpose(np.dot(weights, prevActivations), [1, 0, 2]) + biases
        return z

    def backprop(self, prevActivations, cDerivativeA):
        # prevActivations is shape (N, i, 1)
        # cDerivativeA is shape (N, o, 1)
        z = self.calcZ(self.weights, self.biases, prevActivations)
        if self.activationFunction == "softmax":
            aDerivativeZ = self.softmaxDerivative(z)
        else:
            aDerivativeZ = self.diagMat(self.activationDerivativeZ(z))
        cDerivativeZ = np.matmul(aDerivativeZ, cDerivativeA)
        (zDerivativeW, zDerivativeB, zDerivativePrevA) = self.zDerivativeWBPrevA(prevActivations)
        aDerivativeK = np.diagflat(self.activationDerivativeK(z))
        cDerivativeW = np.mean(np.transpose(np.matmul(zDerivativeW, cDerivativeZ), (0, 2, 1)), 0)
        cDerivativeB = np.mean(np.sum(np.matmul(zDerivativeB, cDerivativeZ), 2, keepdims=True), 0)
        cDerivativeK = np.mean(np.sum(np.matmul(cDerivativeA, aDerivativeK), 2, keepdims=True), 0)
        # Have to diagonalize cDerivativePrevA to be compatiable with calculating the next round of derivatives
        # Will have to undo the diagonalization before unflattening
        cDerivativePrevA = np.sum(np.matmul(zDerivativePrevA, cDerivativeZ), 2, keepdims=True)
        cDerivativePrevA = np.tile(cDerivativePrevA, (1, 1, cDerivativePrevA.shape[1]))
        return cDerivativePrevA, cDerivativeW, cDerivativeB, cDerivativeK

    def zDerivativeWBPrevA(self, prevActivations):
        prevActivations = np.array(prevActivations)
        N = prevActivations.shape[0]
        zDerivativeW = np.tile(prevActivations, (1, 1, self.weights.shape[0]))
        zDerivativePrevA = np.transpose(self.weights, (1, 0))
        zDerivativeB = np.diagflat(np.ones(self.biases.shape))
        return zDerivativeW, zDerivativeB, zDerivativePrevA

    def costDerivativeA(self, idealOutput, actualOutput):
        if self.activationFunction == "softmax":
            return self.crossEntropyDerivative(idealOutput, actualOutput)
        else:
            return self.squaredErrorDerivative(idealOutput, actualOutput)

    @staticmethod
    def crossEntropy(idealOutput, actualOutput):
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cost = -idealOutput * np.log(actualOutput)
        totalCost = sum(cost, 0)
        return cost, totalCost

    @classmethod
    def crossEntropyDerivative(cls,idealOutput, actualOutput):
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cDerivativeA = cls.diagMat(-idealOutput / actualOutput)
        return cDerivativeA

    @staticmethod
    def squaredError(idealOutput, actualOutput):
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cost = np.power(actualOutput - idealOutput, 2)
        totalCost = sum(cost, 0)
        return cost, totalCost

    @classmethod
    def squaredErrorDerivative(cls,idealOutput, actualOutput):
        idealOutput = np.array(idealOutput)
        actualOutput = np.array(actualOutput)
        cDerivativeA = cls.diagMat(2 * (actualOutput - idealOutput))
        return cDerivativeA


class ConvolutionLayer2D:

    # methods to add:
    # create convolution matrix
    # create convolution matrices for backprop

    def __init__(self, imageShape, kernels, biases, stride, ifBuffer):
        # imageShape is (iC, iH, iW)
        # kernels is shape (oC, iC, kH, kW)
        # biases is shape (oC, oH, oW)
        # stride is integer
        # ifBuffer is boolean
        self.kernels = kernels
        self.biases = biases
        self.stride = stride
        self.iC = imageShape[0]
        self.iH = imageShape[1]
        self.iW = imageShape[2]
        self.oC = self.kernels.shape[0]
        self.kH = self.kernels.shape[2]
        self.kW = self.kernels.shape[3]
        if ifBuffer & (self.iW < self.kW | self.iH < self.kH):
            ifBuffer = True
        if ifBuffer:
            if np.remainder(self.kH, 2) != 0:
                self.bufferH = (self.kH - 1)/2
            else:
                self.bufferH = self.kH/2
            if np.remainder(self.kW, 2) != 0:
                self.bufferW = (self.kW - 1)/2
            else:
                self.bufferW = self.kW/2
        else:
            self.bufferH = 0
            self.bufferW = 0
        self.iHB = self.iH + 2*self.bufferH # Input height and width with buffer included
        self.iWB = self.iW + 2*self.bufferW
        self.oHB = np.floor(self.iHB - self.kH) + 1 # Basic output height and width if stride = 1
        self.oWB = np.floor(self.iWB - self.kW) + 1
        self.oH = np.floor((self.iHB - self.kH) / self.stride) + 1
        self.oW = np.floor((self.iWB - self.kW) / self.stride) + 1
        self.createConvolutionMatrix()

    def createConvolutionMatrix(self):
        # Create the convolution matrix for forward propagation
        # self.ConvolutionMatrix is shape (oH*oW*oC, iHB*iWB*iC)
        self.getOutputIndices()
        self.createBasicConvolutionMatrix()
        self.convolutionMatrix = self.basicConvMat[np.newaxis, 0, :]
        for row in np.arange(1, len(self.outputIndexArray)):
            newRow = self.basicConvMat[np.newaxis, self.outputIndexArray[row], :]
            self.convolutionMatrix = np.append(self.convolutionMatrix, newRow, 0)

    def getOutputIndices(self):
        self.outputIndexArray = []
        for channel in np.arange(0, self.oC):
            for r in np.arange(0, self.oHB, self.stride):
                for c in np.arange(0, self.oWB, self.stride):
                    self.outputIndexArray.append(np.ravel_multi_index((channel, r, c), (self.oC, self.oHB, self.oWB)))

    def createBasicConvolutionMatrix(self):
        # self.kernels is shape (oC, iC, kH, kW)
        # basicConvMat is shape (oHB*oWB*oC, iHB*iWB*iC)
        kernels = np.pad(self.kernels, ((0, 0), (0, 0), (0, self.iHB - self.kH), (0, self.iWB - self.kW)))
        self.basicConvMat = np.reshape(kernels, (self.oC, 1, self.iHB * self.iWB * self.iC))
        for r in np.arange(0, self.oWB - 1):
            self.basicConvMat = np.append(self.basicConvMat, np.roll(np.transpose(self.basicConvMat[np.newaxis, :, r, :], (1, 0, 2)), 1), 1)
        for r in np.arange(0, self.oHB - 1):
            self.basicConvMat = np.append(self.basicConvMat, np.roll(self.basicConvMat[:, r * self.oWB:r * self.oWB + self.oWB, :], self.iWB), 1)
        tempMat = self.basicConvMat[0, :, :]
        for channel in np.arange(1, self.oC):
            tempMat = np.append(tempMat, self.basicConvMat[channel, :, :], 0)
        self.basicConvMat = tempMat

    @staticmethod
    def calcOutputShape(imageShape, kernelsShape, stride, ifBuffer):
        # imageShape is (iC, iH, iW)
        # kernelsShape is (oC, iC, kH, kW)
        # stride is integer
        # ifBuffer is boolean
        # Outputs shape of the output without initializing object.
        # Necessary because I need the shape to initialize the kernels and biases and I need the kernels and biases to
        # initialize the layer
        iH = imageShape[1]
        iW = imageShape[2]
        oC = kernelsShape[0]
        kH = kernelsShape[2]
        kW = kernelsShape[3]
        if ifBuffer & (iW < kW | iH < kH):
            ifBuffer = True
        if ifBuffer:
            if np.remainder(kH, 2) != 0:
                bufferH = (kH - 1)/2
            else:
                bufferH = kH/2
            if np.remainder(kW, 2) != 0:
                bufferW = (kW - 1)/2
            else:
                bufferW = kW/2
        else:
            bufferH = 0
            bufferW = 0
        oH = int(np.floor((iH + (2 * bufferH) - kH) / stride) + 1)
        oW = int(np.floor((iW + (2 * bufferW) - kW) / stride) + 1)
        return oC, oH, oW

    def forward(self, inputActivations):
        # inputActivations is shape (N, iC, iH, iW)
        # convolutionMatrix is shape (oC, oH*oW, iHB*iHB*iC)
        # output is shape (N, oC, oH, oW)
        N = inputActivations.shape[0]
        output = np.zeros((N, self.oC, self.oH, self.oW))
        #add buffer
        inputActivations = np.pad(inputActivations, ((0, 0), (0, 0), (self.bufferH, self.bufferH), (self.bufferW, self.bufferW)))
        for n in np.arange(0, N):
            inputSet = np.ravel(inputActivations[n])
            output[n] = np.reshape(np.dot(self.convolutionMatrix, inputSet), (self.oC, self.oH, self.oW))
        output = output + self.biases
        return output

    def backprop(self, prevActivation, cDerivativeZ):
        N = prevActivation.shape[0]
        cDerivativeZBasic = self.cDerivativeZToBasicFormat(cDerivativeZ)
        prevActivation = np.pad(prevActivation, ((0, 0), (0, 0), (self.bufferH, self.bufferH), (self.bufferW, self.bufferW)))
        zDerivativePrevA = self.zDerivativePrevAMatrix()
        zDerivativeW = np.reshape(prevActivation, (N, self.iC * self.iHB * self.iWB))
        cDerivativeB = np.mean(cDerivativeZ, 0)     # I don't need to make it more complicated than this

        # used this as a starting point for the loop to append to, cDerivativePrevA should have 4 dims and cDerivativeW should have 5
        cDerivativePrevA = np.reshape(np.dot(zDerivativePrevA, np.ravel(cDerivativeZBasic[0, :, :, :])), (1, self.iC, self.iHB, self.iWB))
        cDerivativeW = np.reshape(np.dot(self.cDerivativeZMatrix(cDerivativeZBasic[0, :, :, :]), zDerivativeW[0, :]), (1, self.oC, self.iC, self.kH, self.kW))

        # Now to append cDerivativePrevA and cDerivativeW along the N axis
        for n in np.arange(1, N):
            cDerivativePrevA = np.append(cDerivativePrevA, np.reshape(np.dot(zDerivativePrevA, np.ravel(cDerivativeZBasic[n, :, :, :])), (1, self.iC, self.iHB, self.iWB)), 0)
            cDerivativeW = np.append(cDerivativeW, np.reshape(np.dot(self.cDerivativeZMatrix(cDerivativeZBasic[n, :, :, :]), zDerivativeW[n, :]), (1, self.oC, self.iC, self.kH, self.kW)), 0)
        cDerivativeW = np.mean(cDerivativeW, 0) # average dcdw along N axis
        # Remove buffers from cDerivativePrevA
        if self.bufferH > 0:
            cDerivativePrevA = np.delete(cDerivativePrevA, (np.arange(0, self.bufferH), np.arange(self.iHB - self.bufferH, self.iHB)), axis=2)
        if self.bufferW > 0:
            cDerivativePrevA = np.delete(cDerivativePrevA, (np.arange(0, self.bufferW), np.arange(self.iWB - self.bufferW, self.iWB)), axis=3)
        return cDerivativePrevA, cDerivativeW, cDerivativeB

    def cDerivativeZToBasicFormat(self, cDerivativeZ):
        # cDerivativeZ is shape (N, oC, oH, oW)
        cDerivativeZBasic = []
        for n in np.arange(0, cDerivativeZ.shape[0]):
            cDerivativeZBasicSet = np.zeros((1, self.oC*self.oHB*self.oWB))
            cDerivativeZSet = np.ravel(cDerivativeZ[n]) #cDerivativeZSet is shape (oC*oH*oW)
            for r in np.arange(0, len(self.outputIndexArray)):
                cDerivativeZBasicSet[self.outputIndexArray[r]] = cDerivativeZSet[r]
            cDerivativeZBasic = np.append(cDerivativeZBasic, cDerivativeZBasicSet)
        cDerivativeZBasic = np.reshape(cDerivativeZBasic, (cDerivativeZ.shape[0], self.oC, self.oHB, self.oWB))
        return cDerivativeZBasic


        cDerivativeZBase = np.zeros((cDerivativeZ.shape[0], self.oC, self.oHB, self.oWB))
        index = np.unravel_index(self.outputIndexArray, (self.oHB, self.oWB))
        for r in np.arange(0, len(self.outputIndexArray)):
            cDerivativeZBase[:, :, index[r]] = cDerivativeZ[:, :, np.unravel_index(r, (self.oH, self.oW))]
        # cDerivativeZBase is shape (N, oC, oHB, oWB)
        return cDerivativeZBase

    def zDerivativePrevAMatrix(self):
        zDerivativePrevA = np.transpose(self.basicConvMat)
        return zDerivativePrevA

    def cDerivativeZMatrix(self, cDerivativeZ):
        # cDerivativeZ is shape (oC, oHB, oWB)
        # cDerivativeZMat is shape (oC*iC*kH*kW, oC*oHB*oWB)
        # I got to lazy to add the N dim afterwards, will just loop through N
        cDerivativeZ = np.pad(cDerivativeZ, ((0, 0), (0, self.iHB - self.oHB), (0, self.iWB - self.oWB)))
        cDerivativeZ = np.reshape(cDerivativeZ, (self.oC, 1, self.iHB*self.iWB))
        cDerivativeZ = np.pad(cDerivativeZ, ((0, 0), (0, 0), (0, self.iHB*self.iWB)))
        for r in np.arange(0, self.kW - 1):
            cDerivativeZ = np.append(cDerivativeZ, np.roll(np.transpose(cDerivativeZ[np.newaxis, :, r, :], (0, 2, 1, 3)), 1), 2)
        for r in np.arange(0, self.kH - 1):
            cDerivativeZ = np.append(cDerivativeZ, np.roll(cDerivativeZ[:, r * self.kW:r * self.kW + self.kW, :], self.iWB), 2)
        for r in np.arange(0, self.iC -1):
            cDerivativeZ = np.append(cDerivativeZ, np.roll(cDerivativeZ[:, r*self.kW*self.kH:r*self.kW*self.kW + (self.kW*self.kH), :], self.iWB*self.iHB), 2)
        cDerivativeZMat = cDerivativeZ[:, 0, :, :]
        for channel in np.arange(1, self.oC):
            cDerivativeZMat = np.append(cDerivativeZMat, cDerivativeZ[channel, :, :], 1)
        return cDerivativeZMat


class PoolingLayer2D:

    def __init__(self, imageShape, kernelShape, stride):
        # imageShape is (C, iH, iW)
        # kernelShape is (kH, kW)
        # stride is integer
        self.stride = stride
        self.C = imageShape[0]
        self.iH = imageShape[1]
        self.iW = imageShape[2]
        self.kH = kernelShape[0]
        self.kW = kernelShape[1]
        self.oH = np.floor((self.iH - self.kH)/self.stride) + 1
        self.oW = np.floor((self.iW - self.kW)/self.stride) + 1

    @staticmethod
    def calcOutputShape(imageShape, kernelsShape, stride):
        # imageShape is (iC, iH, iW)
        # kernelsShape is (kH, kW)
        # stride is integer
        # Outputs shape of the output without initializing object.
        oC = imageShape[0]
        iH = imageShape[1]
        iW = imageShape[2]
        kH = kernelsShape[0]
        kW = kernelsShape[1]

        oH = int(np.floor((iH - kH) / stride) + 1)
        oW = int(np.floor((iW - kW) / stride) + 1)
        return oC, oH, oW


class MaxPooling2D(PoolingLayer2D):

    def forward(self, image):
        # image is shape (N, C, iH, iW)
        N = image.shape[0]
        output = np.zeros((N, self.C, self.oH, self.oW))
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = image[:, :, y:y + self.kH, x:x + self.kW]
                sectionMax = np.max(imageSection, (2, 3))
                output[:, :, oPosY, oPosX] = sectionMax
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return output

    def backprop(self, prevImage, cDerivativeA):
        # prevImage is shape (N, C, iH, iW)
        # cDerivativeA is shape (N, C, oH, oW)
        N = prevImage.shape[0]
        cDerivativePrevI = np.zeros(prevImage.shape)
        # Loops that get imageSection
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = prevImage[:, :, y:y + self.kH, x:x + self.kW]
                # Loop through N and C bc want max indexes of kHkW for each N and C
                for n in np.arange(0, N):
                    for c in np.arange(0, self.C):
                        maxIndex = np.unravel_inex(np.argmax(imageSection[n, c, :, :]), imageSection[n, c, :, :].shape)
                        cDerivativePrevI[n, c, y + maxIndex[0], x + maxIndex[1]] = cDerivativePrevI[n, c, y + maxIndex[0], x + maxIndex[1]] + cDerivativeA[n, c, oPosY, oPosX]
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return cDerivativePrevI


class MinPooling2D(PoolingLayer2D):

    def forward(self, image):
        # image is shape (N, C, iH, iW)
        N = image.shape[0]
        output = np.zeros((N, self.C, self.oH, self.oW))
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = image[:, :, y:y + self.kH, x:x + self.kW]
                sectionMin = np.min(imageSection, (2, 3))
                output[:, :, oPosY, oPosX] = sectionMin
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return output

    def backprop(self, prevImage, cDerivativeA):
        # prevImage is shape (N, C, iH, iW)
        # cDerivativeA is shape (N, C, oH, oW)
        N = prevImage.shape[0]
        cDerivativePrevI = np.zeros(prevImage.shape)
        # Loops that get imageSection
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = prevImage[:, :, y:y + self.kH, x:x + self.kW]
                # Loop through N and C bc want min indexes of kHkW for each N and C
                for n in np.arange(0, N):
                    for c in np.arange(0, self.C):
                        minIndex = np.unravel_inex(np.argmin(imageSection[n, c, :, :]), imageSection[n, c, :, :].shape)
                        cDerivativePrevI[n, c, y + minIndex[0], x + minIndex[1]] = cDerivativePrevI[n, c, y + minIndex[0], x + minIndex[1]] + cDerivativeA[n, c, oPosY, oPosX]
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return cDerivativePrevI


class MeanPooling2D(PoolingLayer2D):

    def forward(self, image):
        # image is shape (N, C, iH, iW)
        N = image.shape[0]
        output = np.zeros((N, self.C, self.oH, self.oW))
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = image[:, :, y:y + self.kH, x:x + self.kW]
                sectionMean = np.mean(imageSection, (2, 3))
                output[:, :, oPosY, oPosX] = sectionMean
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return output

    def backprop(self, prevImage, cDerivativeA):
        # prevImage is shape (N, C, iH, iW)
        # cDerivativeA is shape (N, C, oH, oW)
        cDerivativePrevI = np.zeros(prevImage.shape)
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                for kPosX in np.arange(0, self.kH):
                    for kPosY in np.arange(0, self.kW):
                        cDerivativePrevI[:, :, y + kPosX, x + kPosY] = cDerivativePrevI[:, :, y + kPosX, x + kPosY] + (cDerivativeA[:, :, oPosY, oPosX]/(self.kH*self.kW))
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return cDerivativePrevI


class L2Pooling2D(PoolingLayer2D):

    def forward(self, image):
        # image is shape (N, C, iH, iW)
        N = image.shape[0]
        output = np.zeros((N, self.C, self.oH, self.oW))
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = image[:, :, y:y + self.kH, x:x + self.kW]
                sectionNorm = np.linalg.norm(imageSection, axis=(2, 3))
                output[:, :, oPosY, oPosX] = sectionNorm
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return output

    def backprop(self, prevImage, cDerivativeA):
        # prevImage is shape (N, C, iH, iW)
        # cDerivativeA is shape (N, C, oH, oW)
        cDerivativePrevI = np.zeros(prevImage.shape)
        oPosY = 0
        for y in np.arange(0, self.iH - self.kH + 1, self.stride):
            oPosX = 0
            for x in np.arange(0, self.iW - self.kW + 1, self.stride):
                imageSection = prevImage[:, :, y:y + self.kH, x:x + self.kW]
                sectionNorm = np.linalg.norm(imageSection, axis=(2, 3))
                for kPosX in np.arange(0, self.kH):
                    for kPosY in np.arange(0, self.kW):
                        cDerivativePrevI[:, :, y + kPosX, x + kPosY] = cDerivativePrevI[:, :, y + kPosX, x + kPosY]*(1 + (cDerivativeA[:, :, oPosY, oPosX]/sectionNorm))
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return cDerivativePrevI


class FlattenLayer2D:

    def __init__(self, inputShape):
        self.inputShape = inputShape

    @staticmethod
    # Flatten 2D
    def forward(image):
        # image is shape (N, iC, iH, iW)
        # output is shape (N, iC*iH*iW, 1)
        output = np.reshape(image, (image.shape[0], np.prod(image.shape[1:]), 1))
        return output

    # Invert Flatten 2D
    def backprop(self, derivativeArray):
        # derivativeArray is shape (N, oC*oH*oW, oC*oH*oW)
        # imageShape is (N, iC, iH, iW)
        # Get rid of diagMat so derivativeArray shape is (N, oC*oH*oW, 1)
        # Then reshape
        derivativeArray = derivativeArray[:, :, 0]
        output = np.reshape(derivativeArray, self.inputShape)
        return output


class InverseFlattenLayer2D:

    def __init__(self, outputShape):
        self.outputShape = outputShape

    # Invert Flatten 2D
    def forward(self, inputActivations):
        # derivativeArray is shape (N, iC*iH*iW, 1)
        # imageShape is (N, iC, iH, iW)
        output = np.reshape(inputActivations, self.outputShape)
        return output

    # Flatten 2D
    def backprop(self, derivativeArray):
        # image is shape (N, iC, iH, iW)
        # output is shape (N, iC*iH*iW, 1)
        output = np.reshape(derivativeArray, (derivativeArray.shape[0], np.prod(derivativeArray.shape[1:]), 1))
        # diagonalize derivative matrices for Fully Connected Layers
        output = np.tile(output, (1, 1, output.shape[1]))
        return output