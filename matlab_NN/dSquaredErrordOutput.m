function dcdo = dSquaredErrordOutput(realOutputs, idealOutputs)
%dSquaredErrordOutput Derivative of the SquaredError function with
%respect to output (aka final layer activation).
%   realOutputs is a nx1 array of activations from the output layer, where
%   n is the number of outputs.
%   idealOutputs is a nx1 array of ideal activations from the output layer
%   corresponding to a specific input.
%   dcdo is a nx1 array resulting from the dSquaredErrordOutput function.
dcdo = 2*(realOutputs-idealOutputs);
end

