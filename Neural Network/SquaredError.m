function cost = SquaredError(realOutputs, idealOutputs)
%SquaredError Loss function comparing real output to ideal output.
%cost = (realOutputs-idealOutputs).^2
%   realOutputs is a nx1 array of activations from the output layer, where
%   n is the number of outputs.
%   idealOutputs is a nx1 array of ideal activations from the output layer
%   corresponding to a specific input.
%   cost is a nx1 array resulting from the SquaredError function.
cost = (realOutputs-idealOutputs).^2;
end

