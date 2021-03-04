function [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,activations0,idealOutputs,oldTotaldCostdWeights,oldTotaldCostdBiases,learningRate,decayFactor,activationFunctionType,outputType)
%GradientStepWithMomentum Like GradientStep but also uses previous dcdw and
%dcdcdb with a decayFactor to add momentum.

%Computes the Gradient
[totaldCostdWeights,totaldCostdBiases] = Gradient(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);

%Takes a step
for layer = 1:numel(weights)
    weights{layer} = weights{layer} - (totaldCostdWeights{layer}*learningRate) + (oldTotaldCostdWeights{layer}*decayFactor);
    biases{layer} = biases{layer} - (totaldCostdBiases{layer}*learningRate) + (oldTotaldCostdBiases{layer}*decayFactor);
end
end

