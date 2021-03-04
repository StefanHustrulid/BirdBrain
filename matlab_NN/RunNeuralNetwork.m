function activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType)
%RunNeuralNetwork Input initial activations, weights, biases, and
%network type to calculate all activations.
%   activations0 Initial activations (aka input)
%   weights Cell array of weight matrices
%   biases Cell array of bias vectors
%   neuralNetworkType Determines which final activation function is used
%   activations All activations in hidden layers and output layer
numberOfLayers = length(biases) + 1;
numberOfHiddenLayers = numberOfLayers - 2;
activations = cell(1,numberOfHiddenLayers+1);
activations{1} = ActivationFunction(activations0,weights{1},biases{1},activationFunctionType);
if numberOfHiddenLayers > 1
    for layer = 2:numberOfLayers-2
        activations{layer} = ActivationFunction(activations{layer-1},weights{layer},biases{layer},activationFunctionType);
    end
end
lastLayer = numberOfLayers-1;
activations{lastLayer} = FinalActivationFunction(activations{lastLayer-1}, weights{lastLayer}, biases{lastLayer}, outputType);
end

function output = FinalActivationFunction(lastHiddenActivation, lastWeights, lastBiases, outputType)
if outputType == 1
    output = SoftmaxActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 2
    output = TanhActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 3
    output = LinearActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 4
    output = SigmoidActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 5
    output = SoftplusActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 6
    output = -SigmoidActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
elseif outputType == 7
    output = -SoftplusActivationFunction(lastHiddenActivation, lastWeights, lastBiases);
end
end

