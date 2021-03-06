function dadz = dLeakyReLUdz(previousActivations,layerWeights,layerBiases)
%dLeakyReLUdz Derivative of the LeakyReLU activation function with respect to
%z, where z = (W*a)+b.
%   previousActivation is an nx1 array of the activations from the previous 
%   layer, where n is the number of neurons in the previous layer.
%   layerWeights is an mxn matrix of weights, where m is the number of 
%   neurons in the current layer.
%   layerBiases is an mx1 array of the biases in the current layer.
%   dadz is an mx1 array of the  derivative of the activation of the 
%   current layer with respect to z.
constant = 0.1; %some constant between 0&1
z = (layerWeights*previousActivations) + layerBiases;
dadz = (z >= 0) + constant*(z < 0);
end


