function activation = SELUActivationFunction(previousActivation, layerWeights, layerBiases)
%SELUActivationFunction Takes in previous layer activations, current
%weights and current biases and results in current layer activation using 
%the SELU function.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   acivation is an mx1 array of the activation of the current layer.

alphaConstant = 1.6732632423543772848170429916717;
lambdaConstant = 1.0507009873554804934193349852946;

z = (layerWeights*previousActivation) + layerBiases;
activation = lambdaConstant*((z.*(z >= 0)) + ((alphaConstant*exp(z)-alphaConstant).*(z < 0)));
end
