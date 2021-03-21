function [weights,biases,newWeightChanges,newBiasChanges] = GradientStepWithMomentum(weights,biases,activations0,idealOutputs,oldWeightChanges,oldBiasChanges,learningRate,decayFactor,activationFunctionType,outputType)
%GradientStepWithMomentum Like GradientStep but also uses previous dcdw and
%dcdcdb with a decayFactor to add momentum.

%Computes the Gradient
[totaldCostdWeights,totaldCostdBiases] = Gradient(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);

oldWeights = cell(1,numel(weights));
oldBiases = cell(1,numel(biases));
newWeightChanges = cell(1,numel(oldWeightChanges));
newBiasChanges = cell(1,numel(oldBiasChanges));


%Takes a step
for layer = 1:numel(weights)
    oldWeights{layer} = weights{layer};
    weights{layer} = weights{layer} - (totaldCostdWeights{layer}*learningRate) + (oldWeightChanges{layer}*decayFactor);
    newWeightChanges{layer} = weights{layer} - oldWeights{layer};
    
    oldBiases{layer} = biases{layer};
    biases{layer} = biases{layer} - (totaldCostdBiases{layer}*learningRate) + (oldBiasChanges{layer}*decayFactor);
    newBiasChanges{layer} = biases{layer} - oldBiases{layer};
end
end

