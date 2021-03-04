function dadz = dTanhdz(previousActivations,layerWeights,layerBiases)
%dTanhdz Derivative of the Tanh activation function with respect to
%z, where z = (W*a)+b.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   dadz is an mx1 array of the  derivative of the activation of the 
%   current layer with respect to z.
activations = TanhActivationFunction(previousActivations,layerWeights,layerBiases);
dadz = 1-(activations.^2);
end

