function costs = Cost(realOutputs, idealOutputs, outputType)
%Outputs costs array when given realOutputs, idealOutputs, and
%neuralNetworkType
if outputType == 1
    costs = CrossEntropy(realOutputs, idealOutputs);
else
    costs = SquaredError(realOutputs, idealOutputs);
end
end
