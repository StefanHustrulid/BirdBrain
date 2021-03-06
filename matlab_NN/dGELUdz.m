function dadz = dGELUdz(previousActivations,layerWeights,layerBiases)
%dGELUdz Derivative of the GELU activation function with respect to
%z, where z = (W*a)+b.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   dadz is an mx1 array of the  derivative of the activation of the 
%   current layer with respect to z.

z = (layerWeights*previousActivations) + layerBiases;
dadz = 0.5*tanh((0.0356774*z.^3)+(0.797885*z))  +(((0.0535161*z.^3)+(0.398942*z)).*((sech((0.0356774*z.^3)+(0.797885*z))).^2)) + 0.5;
end