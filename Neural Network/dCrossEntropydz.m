function dcdz = dCrossEntropydz(realOutputs, idealOutputs)
%dCrossEntropydz Derivative of the CrossEntropy function with
%respect to z (Wa+b before activation).
%   realOutputs is a nx1 array of activations from the output layer, where
%   n is the number of outputs.
%   idealOutputs is a nx1 array of ideal activations from the output layer
%   corresponding to a specific input.
%   dcdz is a nx1 array resulting from the dCrossEntropydz function.
dcdz = realOutputs-idealOutputs;
end

