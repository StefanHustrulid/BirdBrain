function [costs,totalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType)
%TestNeuralNetwork Calculates the array costs and total costs of the
%current neural network for a single dataset
numberOfLayers = numel(biases)+1;
numberOfOutputs = length(idealOutputs(:,1));
costs = zeros(numberOfOutputs,1);
for constraint = 1:size(activations0(1,:))
    activations = RunNeuralNetwork(activations0(:,constraint),weights,biases,activationFunctionType,outputType);
    costs = costs + (Cost(activations{numberOfLayers-1},idealOutputs(:,constraint),outputType)/length(activations0(1,:)));
end
totalCost = sum(costs);
end

