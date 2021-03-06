function dadz = dSELUdz(previousActivations,layerWeights,layerBiases)
%dSELUdz Derivative of the SELU activation function with respect to
%z, where z = (W*a)+b.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   dadz is an mx1 array of the  derivative of the activation of the 
%   current layer with respect to z.

alphaConstant = 1.6732632423543772848170429916717;
lambdaConstant = 1.0507009873554804934193349852946;

z = (layerWeights*previousActivations) + layerBiases;
dadz = lambdaConstant*((z >= 0) + alphaConstant*exp(z).*(z < 0));
end