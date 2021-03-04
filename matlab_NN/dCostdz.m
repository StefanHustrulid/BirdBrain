function dcdz = dCostdz(realOutputs, idealOutputs, lastHiddenActivation, lastWeights, lastBiases, outputType)
%dCostcz Calculates dcdz of last layer depending on neuralNetworkType.
if outputType == 1
    dcdz = realOutputs-idealOutputs;
elseif outputType == 2
    tanH = TanhActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = 2*(tanH-idealOutputs).*dTanhdz(lastHiddenActivation,lastWeights,lastBiases);
elseif outputType == 3
    linear = LinearActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = 2*(linear-idealOutputs).*dLinearz(lastHiddenActivation,lastWeights,lastBiases);
elseif outputs == 4
    sigmoid = SigmoidActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = 2*(sigmoid-idealOutputs).*dSigmoiddz(lastHiddenActivation,lastWeights,lastBiases);
elseif outputs == 5
    softplus = SoftplusActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = 2*(softplus-idealOutputs).*dSoftplusdz(lastHiddenActivation,lastWeights,lastBiases);
elseif outputs == 6
    sigmoid = SigmoidActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = -2*(sigmoid-idealOutputs).*dSigmoidhdz(lastHiddenActivation,lastWeights,lastBiases);
elseif outputs == 7
    softplus = SoftplusActivationFunction(lastHiddenActivation,lastWeights,lastBiases);
    dcdz = -2*(softplus-idealOutputs).*dSoftplusdz(lastHiddenActivation,lastWeights,lastBiases);
end