function activation = ActivationFunction(previousActivation,layerWeights,layerBiases,activationFunctionType)
%ActivationFunction Manages which activation functions are used.
if activationFunctionType == 1
    activation = LinearActivationFunction(previousActivation,layerWeights,layerBiases);
elseif activationFunctionType == 2
    activation = SigmoidActivationFunction(previousActivation,layerWeights,layerBiases);
elseif activationFunctionType == 3
    activation = TanhActivationFunction(previousActivation,layerWeights,layerBiases);
elseif activationFunctionType == 4
    activation = ReLUActivationFunction(previousActivation,layerWeights,layerBiases);
end
end

