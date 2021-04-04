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


class NN:
    # Neural Network common methods:
    # Activation Functions and their derivatives
    x = "Hi"



class ActivationLayer:

    def __init__(self, constants, activationFunction):
        self.constants = constants
        self.activationFunction = activationFunction

    def activation(self, z):
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
        z = np.array(z)
        activation = z
        return activation

    @staticmethod
    def linearDerivative(z):
        z = np.array(z)
        aDerivativeZ = np.ones(z.shape)
        return aDerivativeZ

    @staticmethod
    def sigmoidActivation(z):
        z = np.array(z)
        activation = 1 / (1 + np.exp(-z))
        return activation

    @staticmethod
    def sigmoidDerivative(z):
        z = np.array(z)
        activation = ActivationLayer.sigmoidActivation(z)
        aDerivativeZ = activation * (1 - activation)
        return aDerivativeZ

    @staticmethod
    def tanhActivation(z):
        z = np.array(z)
        activation = np.tanh(z)
        return activation

    @staticmethod
    def tanhDerivative(z):
        z = np.array(z)
        aDerivativeZ = 1 - np.power(z, 2)
        return aDerivativeZ

    @staticmethod
    def ReLUActivation(z):
        z = np.array(z)
        activation = z * (z >= 0)
        return activation

    @staticmethod
    def ReLUDerivative(z):
        z = np.array(z)
        aDerivativeZ = 1 * (z >= 0)
        return aDerivativeZ

    @staticmethod
    def LReLUActivation(z, constant):
        # constant is a single float from 0 to 1
        z = np.array(z)
        activation = (z * (z >= 0)) + (constant * z * (z < 0))
        return activation

    @staticmethod
    def LReLUDerivative(z, constant):
        # constant is a single float from 0 to 1
        z = np.array(z)
        aDerivativeZ = (1 * (z >= 0)) + (constant * (z < 0))
        return aDerivativeZ

    @staticmethod
    def PReLUActivation(z, constant):
        # constant is an array of floats from 0-1 same dim as z
        z = np.array(z)
        constant = np.array(constant)
        activation = (z * (z >= 0)) + (constant * z * (z < 0))
        return activation

    @staticmethod
    def PReLUDerivative(z, constant):
        # constant is an array of floats from 0-1 same dim as z
        z = np.array(z)
        constant = np.array(constant)
        aDerivativeZ = (1 * (z >= 0)) + (constant * (z < 0))
        aDerivativeK = z * (z < 0)
        return aDerivativeZ, aDerivativeK

    @staticmethod
    def ELUActivation(z, constant):
        # constant is a single float > 0
        z = np.array(z)
        activation = (z * (z >= 0)) + (constant * (np.exp(z) - 1) * (z < 0))
        return activation

    @staticmethod
    def ELUDerivative(z, constant):
        # constant is a single float > 0
        z = np.array(z)
        aDerivativeZ = (1 * (z >= 0)) + (constant * np.exp(z) * (z < 0))
        return aDerivativeZ

    @staticmethod
    def SELUActivation(z):
        alphaConstant = 1.6732632423543772848170429916717
        lambdaConstant = 1.0507009873554804934193349852946
        z = np.array(z)
        activation = lambdaConstant * z * ((z >= 0) + (alphaConstant * (np.exp(z) - 1) * (z < 0)))
        return activation

    @staticmethod
    def SELUDerivative(z):
        alphaConstant = 1.6732632423543772848170429916717
        lambdaConstant = 1.0507009873554804934193349852946
        z = np.array(z)
        aDerivativeZ = lambdaConstant * ((z >= 0) + (alphaConstant * np.exp(z) * (z < 0)))
        return aDerivativeZ

    @staticmethod
    def GELUActivtion(z):
        z = np.array(z)
        activation = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + (0.044715 * np.power(z, 3)))))
        return activation

    @staticmethod
    def GELUDerivative(z):
        z = np.array(z)
        aDerivativeZ = (0.5 * np.tanh((0.0356774 * np.power(z, 3)) + (0.797885 * z))) + (
                    ((0.535161 * np.power(z, 3)) + (0.398942 * z)) * np.power(
                np.sech((0.0356774 * np.power(z, 3)) + (0.797885 * z)), 2)) + 0.5
        return aDerivativeZ

    @staticmethod
    def softsignActivation(z):
        z = np.array(z)
        activation = z / (np.abs(z) + 1)
        return activation

    @staticmethod
    def softsignDerivative(z):
        z = np.array(z)
        absolute = np.abs(z)
        aDerivativeZ = ((z >= 0) * (absolute + 1 - z) / np.power(np.abs(z) + 1, 2)) + (
                    (z < 0) * (np.abs(z) + 1 + z) / np.power(absolute + 1, 2))
        return aDerivativeZ

    @staticmethod
    def softplusActivation(z):
        z = np.array(z)
        activation = np.log(1 + np.exp(z))
        return activation

    @classmethod
    def softplusDerivative(cls,z):
        z = np.array(z)
        aDerivativeZ = cls.sigmoidActivation(z)
        return aDerivativeZ

    @staticmethod
    def softmaxActivation(z):
        z = np.array(z)
        activation = z / np.sum(z, 0)
        return activation

    @classmethod
    def softmaxDerivative(cls, z):
        z = np.array(z)
        activation = cls.softmaxActivation(z)
        diagMatrix = cls.diagMat(activation)
        softmaxMatrix = np.tile(activation, (1, 1, 2))
        aDerivativeZ = diagMatrix - np.matmul(softmaxMatrix, np.transpose(softmaxMatrix, (0, 2, 1)))
        return aDerivativeZ

    @staticmethod
    def diagMat(inputStack):
        # Custom method that takes (N, m, 1) or (N, 1, m) array
        # And outputs (n, m, m) diagonal matrices
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
        if ifBuffer == False & (self.iW < self.kW | self.iH < self.kH):
            ifBuffer = True
        if ifBuffer == True:
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

    def convoluteLayer(self, image):
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


class MaxPooling2D(PoolingLayer2D):

    def maxPooling(self, image):
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

    def minPooling(self, image):
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

    def meanPooling(self, image):
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

    def L2Pooling(self, image):
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

    @staticmethod
    def flatten2D(image):
        # image is shape (N, iC, iH, iW)
        # output is shape (N, iC*iH*iW, 1)
        output = np.reshape(image, (image.shape[0], np.prod(image.shape[1:]), 1))
        return output

    @staticmethod
    def invertFlatten2D(derivativeArray, imageShape):
        # Make sure to use ActivationLayer.invDiagMat() First!
        # derivativeArray is shape (N, iC*iH*iW, 1)
        # imageShape is (N, iC, iH, iW)
        output = np.reshape(derivativeArray, imageShape)
        return output


class ANN(NN):

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
