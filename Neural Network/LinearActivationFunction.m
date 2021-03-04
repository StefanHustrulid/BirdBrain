function activation = LinearActivationFunction(previousActivation, layerWeights, layerBiases)
%LinearActivationFunction Takes in previous layer activations, current
%weights and current biases and results in current layer activation using 
%the Linear function z = Wa + b.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   acivation is an mx1 array of the activation of the current layer.
activation = (layerWeights*previousActivation) + layerBiases;
end

