function cost = CrossEntropy(realOutputs, idealOutputs)
%CrossEntropy Loss function comparing real output to ideal output.
%Designed for softmax output.
%cost = -idealOutputs.*log(realOutputs)
%   realOutputs is a nx1 array of activations from the output layer, where
%   n is the number of outputs.
%   idealOutputs is a nx1 array of ideal activations from the output layer
%   corresponding to a specific input.
%   cost is a nx1 array resulting from the CrossEntropy function.
cost = -idealOutputs.*log(realOutputs);
end

