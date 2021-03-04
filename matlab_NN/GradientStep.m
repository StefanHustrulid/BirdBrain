function [weights,biases] = GradientStep(weights,biases,activations0,idealOutputs,learningRate,activationFunctionType,outputType)
%GradientStep Improves the Weights and Biases by 1 step

%Computes the Gradient
[totaldCostdWeights,totaldCostdBiases] = Gradient(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);

%Takes a step
for layer = 1:numel(weights)
    weights{layer} = weights{layer} - (totaldCostdWeights{layer}*learningRate);
    biases{layer} = biases{layer} - (totaldCostdBiases{layer}*learningRate);
end
end

