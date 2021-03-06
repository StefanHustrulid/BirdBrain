function activation = SoftmaxActivationFunction(previousActivation, layerWeights, layerBiases)
%SoftmaxActivationFunction Takes in previous layer activations, current
%weights and current biases and results in current layer activation using 
%the Softmax function.
%Use on the output layer if a classification output is desired. The sum of the
%activations on the layer is 1
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   acivation is an mx1 array of the activation of the current layer.
z = (layerWeights*previousActivation)+layerBiases;
activation = exp(z)./sum(exp(z)); 
end

