function dcdz = dSquaredErrordz(realOutputs, idealOutputs, lastHiddenActivation, lastWeights, lastBiases)
%dSquarredErrordz Derivative of the SquaredError function with
%respect to z (Wa+b before activation).
%   realOutputs is a nx1 array of activations from the output layer, where
%   n is the number of outputs.
%   idealOutputs is a nx1 array of ideal activations from the output layer
%   corresponding to a specific input.
%   dcdz is a nx1 array resulting from the dSquarredErrordz function.
z = (lastWeights*lastHiddenActivation)+lastBiases;
dcdz = -2*(e^(-z)).*(realOutputs.^2).*(realOutputs-idealOutputs);
end

