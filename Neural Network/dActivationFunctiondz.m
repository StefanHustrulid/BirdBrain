function dadz = dActivationFunctiondz(previousActivations,layerWeights,layerBiases,activationFunctionType)
%dActivationFunctiondz Manages which activation functions are used to
%calculate dadz.

if activationFunctionType == 1
    dadz = dLineardz(previousActivations,layerWeights,layerBiases);
elseif activationFunctionType == 2
    dadz = dSigmoiddz(previousActivations,layerWeights,layerBiases);
elseif activationFunctionType == 3
    dadz = dTanhdz(previousActivations,layerWeights,layerBiases);
elseif activationFunctionType == 4
    dadz = dReLUdz(previousActivations,layerWeights,layerBiases);
end
end

