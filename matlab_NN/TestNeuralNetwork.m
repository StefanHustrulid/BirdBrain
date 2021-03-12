function [costs,totalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType)
%TestNeuralNetwork Calculates the array costs and total costs of the
%current neural network for a single dataset

activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);
costs = mean(Cost(activations{end},idealOutputs,outputType),2);
totalCost = sum(costs);
end

