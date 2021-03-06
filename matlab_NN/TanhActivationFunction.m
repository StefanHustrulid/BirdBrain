function activation = TanhActivationFunction(previousActivation, layerWeights, layerBiases)
%TanhActivationFunction Takes in previous layer activations, current
%weights and current biases and results in current layer activation using 
%the Tanh function.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   acivation is an mx1 array of the activation of the current layer.
z = (layerWeights*previousActivation) + layerBiases;
activation = tanh(z);
end
