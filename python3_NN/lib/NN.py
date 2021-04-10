## Variables:
#
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

class NeuralNetwork:

    def __init__(self, isNew):
        # isNew is boolean
        NeuralNetworkStructure = []
        NeuralNetwork = []
        if isNew == False:
            filename = input("Neural Network Structure File: ")
            NNSFile = json.open(filename)
            NeuralNetworkStructure = json.load(NNSFile)
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
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, "Linear", distribution)
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
                    NeuralNetworkStructure[layer]["Biases"] = biases
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
                        inputShape = [iH, 1]
                    else:
                        inputShape = NeuralNetworkStructure[layer - 1]["Output Shape"]

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
                    NeuralNetworkStructure[layer]["Constants"] = constants
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
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, activationFunction, distribution)
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
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, activationFunction, distribution)
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
                    NeuralNetworkStructure[layer]["Weights"] = weights
                    NeuralNetworkStructure[layer]["Biases"] = biases
                    NeuralNetworkStructure[layer]["Constants"] = constants
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
                            NeuralNetworkStructure[layer - l]["Kernels"] = ParameterInitializer.initializeWeights(fanIn, fanOut, kernelShape, "Linear", distribution)
                            print("Layer", layer - l, "Kernels Initialized")
                            break
                    for l in np.arange(0, layer):
                        print("\nLayer", l,"Information:")
                        for entry in NeuralNetworkStructure[l].keys():
                            print(entry, ":", NeuralNetworkStructure[l][entry])
                    break
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
                NeuralNetwork.append(InvertFlattenLayer2D(outputShape))
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

    def runNeuralNetwork(self, inputActivations):
        # inputActivations must be a list of arrays with shape (N, iH, 1) where N is constant and iH may be variable
        # ex: [[[[1], [2], [3]], [[11], [12], [13]]], [[[4], [5]], [[14], [15]]]]
        # ex: [[[[1], [2], [3]], [[11], [12], [13]]]]
        # ex: [[[[1], [2], [3]]], [[[4], [5]]]]
        # ex: [[[[1], [2], [3]]]]
        inputSet = 0
        activations = [np.array(inputActivations[inputSet])]
        for layer in np.arange(0, np.size(self.NeuralNetwork)):
            if layer > 0:
                if self.NeuralNetworkStructure[layer]["Layer Type"] == "Fully Connected":
                    if self.NeuralNetworkStructure[layer]["Input Size"] > self.NeuralNetworkStructure[layer - 1]["Output Size"]:
                        inputSet = inputSet + 1
                        activations[layer] = np.append[activations[layer], inputActivations[inputSet, 1]]
            activations.append(self.NeuralNetwork[layer].forward(activations[layer]))
        return activations

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

    @staticmethod
    def sigmoidDerivative(z):
        # z is any shape
        # output is same shape as z
        z = np.array(z)
        activation = ActivationLayer.sigmoidActivation(z)
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
        aDerivativeZ = (0.5 * np.tanh((0.0356774 * np.power(z, 3)) + (0.797885 * z))) + (
                    ((0.535161 * np.power(z, 3)) + (0.398942 * z)) * np.power(
                np.sech((0.0356774 * np.power(z, 3)) + (0.797885 * z)), 2)) + 0.5
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
        aDerivativeZ = ((z >= 0) * (absolute + 1 - z) / np.power(np.abs(z) + 1, 2)) + (
                    (z < 0) * (np.abs(z) + 1 + z) / np.power(absolute + 1, 2))
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
        diagMatrix = np.array([])
        for N in inputStack:
            nDiagMatrix = np.diagflat(N)
            diagMatrix = np.stack((diagMatrix, nDiagMatrix)) if diagMatrix.size else nDiagMatrix
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
        self.weights = weights
        self.biases = biases
        self.constants = constants
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
        z = np.dot(weights * prevActivations) + biases
        return z

    def layerGradient(self, prevActivations, cDerivativeA):
        # prevActivations is shape (N, i, 1)
        # cDerivativeA is shape (N, o, 1)

        z = self.calcZ(self.weights, self.biases, prevActivations)

        if self.activationFunction == "softmax":
            aDerivativeZ = self.softmaxDerivative(z)
        else:
            aDerivativeZ = self.diagMat(self.activationDerivativeZ(z))

        cDerivativeZ = np.matmul(aDerivativeZ, cDerivativeA)

        (zDerivativeW, zDerivativeB, zDerivativePrevA) = self.zDerivativeWBPrevA(prevActivations)

        aDerivativeK = self.diagMat(self.activationDerivativeK(z))

        cDerivativeW = np.mean(np.transpose(np.matmul(zDerivativeW, cDerivativeZ), (0, 2, 1)), 0)
        cDerivativeB = np.mean(np.sum(np.matmul(zDerivativeB, cDerivativeZ), 2, keepdims=True), 0)
        cDerivativeK = np.mean(np.sum(np.matmul(cDerivativeA, aDerivativeK), 2, keepdims=True), 0)
        # Have to diagonalize cDerivativePrevA to be compatiable with calculating the next round of derivatives
        # Will have to undo the diagonalization before unflattening
        cDerivativePrevA = self.diagMat(np.sum(np.matmul(zDerivativePrevA, cDerivativeZ), 2, keepdims=True))
        return cDerivativeW, cDerivativeB, cDerivativeK, cDerivativePrevA

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
        self.oH = np.floor((self.iH + (2 * self.bufferH) - self.kH) / self.stride) + 1
        self.oW = np.floor((self.iW + (2 * self.bufferW) - self.kW) / self.stride) + 1

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

    def forward(self, image):
        # image is shape (N, iC, iH, iW)
        N = image.shape[0]
        output = np.zeros((N, self.oC, self.oH, self.oW))
        #add buffer
        image2 = np.pad(image, ((0, 0), (0, 0), (self.bufferH, self.bufferH), (self.bufferW, self.bufferW)))
        # tile image for all kernel/output channels and transpose to (NxoCxiCxiHxiW)
        image3 = np.transpose(np.tile(image2, (self.oC, 1, 1, 1, 1)), (1, 0, 2, 3, 4))
        oPosY = 0
        for y in np.arange(self.bufferH, self.iH + self.bufferH, self.stride):
            oPosX = 0
            for x in np.arange(self.bufferW, self.iW + self.bufferH, self.stride):
                # kHxkW section of iHxiW of the image for all N, iC, and oC
                imageSection = image3[:, :, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1]
                # Element-wise multiplication then sum along the iC, kH, and kW axis to get output at that XY for each N and oC
                k = np.sum(imageSection*self.kernels, (2, 3, 4))
                output[:, :, oPosY, oPosX] = k
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        # Kernels acted as weights now to add the biases
        # output of convolutional layer is equivalent to z
        output = output + self.biases
        return output

    def layerGradient(self, prevImage, cDerivativeZ):
        # prevImage is shape (N, iC, iH, iW)
        # cDerivativeZ is shape (N, oC, oH, oW)
        (zDerivativeW, zDerivativeB, zDerivativePrevA) = self.zDerivativeWBPrevA(prevImage)
        # zDerivativeW is shape (N, iC, kH, kW)
        # zDerivativeB is shape (oC, oH, oW)
        # zDerivativePrevA is shape (N, oC, iC, iH+2bH, iW+2bW)

        # Transpose zDerivativePrevA to (iC, N, oC, iH+2bH, iW+2BW) to be compatible with cDerivativeZ
        zDerivativePrevA2 = np.transpose(zDerivativePrevA, (2, 0, 1, 3, 4))

        # cDerivativeW is shape (oC, kH, kW)
        # cDerivativeB is shape (oC, oH, oW)
        # cDerivativePrevA is shape (N, iC, iH, iW)

        # To get dcdpa I need to multiply dzdpa with dcdz at corresponding channels and locations, then sum along oC axis
        # To get dcdw I need to multiply dzdw with dcdz at corresponding channels and locations, then avg along N axis
        # To get dcdb I need to multiply dcdb with dcdz and avg along N axis

        cDerivativeB = np.mean(zDerivativeB*cDerivativeZ, 0)

        # add buffer
        prevImage2 = np.pad(prevImage, ((0, 0), (0, 0), (self.bufferH, self.bufferH), (self.bufferW, self.bufferW)))

        cDerivativePrevA = np.zeros(prevImage2.shape)
        cDerivativeW = np.zeros(self.kernels.shape)
        oPosY = 0
        for y in np.arange(self.bufferH, self.iH + self.bufferH, self.stride):
            oPosX = 0
            for x in np.arange(self.bufferW, self.iW + self.bufferH, self.stride):
                # This is very complicated to explain
                # multiply the section of zDerivativePrevA with the corresponding cDerivativeZ
                # then sum along the oC axis
                # then transpose to be compatible with cDerivativeA
                # then add to the cDerivativeA section
                cDerivativePrevA[:, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1] = cDerivativePrevA[:, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1] + np.transpose(np.sum(zDerivativePrevA[:, :, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1]*cDerivativeZ[:, :, oPosY, oPosX], 2), (1, 0, 2, 3))

                # Multiply the zDerivativeW with the corresponding cDerivativeZ
                # then average along the N axis and add to the cDerivativeW
                cDerivativeW = cDerivativeW + np.mean(zDerivativeW*cDerivativeZ[:, :, oPosY, oPosX], 0)
                oPosX = oPosX + 1
            oPosY = oPosY + 1

        # Remove buffer edges from cDerivativePrevA
        if self.bH > 0:
            cDerivativePrevA = np.delete(cDerivativePrevA, (np.arange(0, self.bH), np.arange(self.bH + self.iH, self.bH + self.iH + self.bH)), axis=3)
        if self.bW > 0:
            cDerivativePrevA = np.delete(cDerivativePrevA, (np.arange(0, self.bW), np.arange(self.bW + self.iW, self.bW + self.iW + self.bW)), axis=4)
        return cDerivativeW, cDerivativeB, cDerivativePrevA

    def zDerivativeWBPrevA(self, prevImage):
        # prevImage is shape (N, iC, iH, iW)
        # zDerivativeW is shape (N, iC, kH, kW)
        # zDerivativeB is shape (oC, oH, oW)
        # cDerivativePrevA is shape (N, iC, oC, iH+2bH, iW+2bW)
        N = prevImage.shape[0]
        zDerivativeW = np.zeros((N, self.iC, self.kH, self.kW))
        zDerivativeB = np.ones((self.oC, self.oH, self.oW))

        # add buffer
        prevImage2 = np.pad(prevImage, ((0, 0), (0, 0), (self.bufferH, self.bufferH), (self.bufferW, self.bufferW)))
        # tile image for all kernel/output channels and transpose to (NxoCxiCxiHxiW)
        zDerivativePrevA = np.transpose(np.tile(np.zeros(prevImage2.shape), (self.oC, 1, 1, 1, 1)), (1, 0, 2, 3, 4))

        oPosY = 0
        for y in np.arange(self.bufferH, self.iH + self.bufferH, self.stride):
            oPosX = 0
            for x in np.arange(self.bufferW, self.iW + self.bufferH, self.stride):
                # add the weights from the kernels to their corresponding spot on zDerivativePrevA
                zDerivativePrevA[:, :, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1] = zDerivativePrevA[:, :, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1] + self.kernels
                # kHxkW section of iHxiW of the image for all N, iC, and oC
                imageSection = prevImage2[:, :, y - self.bufferH:y + self.bufferH + 1, x - self.bufferW:x + self.bufferW + 1]
                zDerivativeW = zDerivativeW + imageSection
                oPosX = oPosX + 1
            oPosY = oPosY + 1
        return zDerivativeW, zDerivativeB, zDerivativePrevA



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

    def maxPoolingBackpropogation(self, prevImage, cDerivativeA):
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

    def minPoolingBackpropogation(self, prevImage, cDerivativeA):
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

    def meanPoolingBackpropogation(self, prevImage, cDerivativeA):
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

    def L2PoolingBackpropogation(self, prevImage, cDerivativeA):
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
    def invertFlatten2D(self, derivativeArray):
        # derivativeArray is shape (N, oC*oH*oW, oC*oH*oW)
        # imageShape is (N, iC, iH, iW)
        # Get rid of diagMat so derivativeArray shape is (N, oC*oH*oW, 1)
        # Then reshape
        derivativeArray = self.invDiagMat(derivativeArray)
        output = np.reshape(derivativeArray, self.inputShape)
        return output

    @staticmethod
    def invDiagMat(diagMatStack):
        # Custom method that takes diagonal matrices with shape (N, m, m)
        # And outputs array with shape (N, m, 1)
        diagMatStack = np.array(diagMatStack)
        outputArray = np.sum(diagMatStack, 2, keepdims=True)
        return outputArray


class InvertFlattenLayer2D:

    def __init__(self, outputShape):
        self.outputShape = outputShape

    # Invert Flatten 2D
    def forward(self, inputActivations):
        # derivativeArray is shape (N, iC*iH*iW, 1)
        # imageShape is (N, iC, iH, iW)
        output = np.reshape(inputActivations, self.outputShape)
        return output

    # Flatten 2D
    def flatten2D(self, derivativeArray):
        # image is shape (N, iC, iH, iW)
        # output is shape (N, iC*iH*iW, 1)
        output = np.reshape(derivativeArray, (derivativeArray.shape[0], np.prod(derivativeArray.shape[1:]), 1))
        # diagonalize derivative matrices for Fully Connected Layers
        output = self.diagMat(output)
        return output

    @staticmethod
    def diagMat(inputStack):
        # Custom method that takes (N, m, 1) or (N, 1, m) array
        # And outputs (N, m, m) diagonal matrices
        inputStack = np.array(inputStack)
        diagMatrix = np.array([])
        for N in inputStack:
            nDiagMatrix = np.diagflat(N)
            diagMatrix = np.stack((diagMatrix, nDiagMatrix)) if diagMatrix.size else nDiagMatrix
        return diagMatrix



class ANN:

    #
    # Init Function for Artificial Neural Network.
    # var load: true to load from file, false to create new
    #
    def __init__(self, load):
        if (load == True):
            #Load the artificial neural network from a file with predetermined structure
            print("Loading from file...")

        else:
            # Initialize Artificial Neural Net layers
            self.layers = []
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
            menu['6'] = "ELU"
            menu['7'] = "SELU"
            menu['8'] = "GELU"
            for entry in menu.keys():
                print(entry, menu[entry])
            selection = input("Select an activation function: ")
            self.activationType = menu[selection]
            print("Activation function: ", self.activationType, "\n")

            # Set output type
            menu = {}
            menu['1'] = "0 to 1"
            menu['2'] = "-1 to 0"
            menu['3'] = "0 to inf"
            menu['4'] = "-inf to 0"
            menu['5'] = "-inf to inf"
            menu['6'] = "0 to 100%"
            for entry in menu.keys():
                print(entry, menu[entry])
            selection = input("Select an output type: ")
            self.outputType = menu[selection]
            print("Output type: ", self.outputType, "\n")

            # Set Weight initialization method
            self.weights = []
            self.biases = []
            self.lastWeightDelta = []
            self.lastBiasDelta = []
            self.learningRate = 0.01
            self.decayRate = 0.5
            menu = {}
            menu['1'] = "Uniform Xavier"
            menu['2'] = "Gaussian Xavier"
            menu['3'] = "Uniform He"
            menu['4'] = "Gaussian He"
            menu['5'] = "Uniform LeCun"
            menu['6'] = "Gaussian LeCun"
            if (self.activationType == "Linear") or (self.activationType == "Sigmoid") or (self.activationType == "Tanh"):
                print("Recommended weight initialization method is Xavier/Glorot")
            elif (self.activationType == "ReLU") or (self.activationType == "Leaky ReLU"):
                print("Recommended weight initialization method is He/Kaiming")
            else:
                print("Recommended weight initialization method is LeCun")
            for entry in menu.keys():
                print(entry, menu[entry])
            selection = input("Select your preferred Weight initialization method: ")
            self.weightInitMethod = menu[selection]
            print("Weight initialization method: ", self.weightInitMethod, "\n")
            # As far as I got
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
            dcda = -1*(ideal/output)
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
