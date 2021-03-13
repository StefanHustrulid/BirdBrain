function [dzdw,dzdb,dzdaprev] = dzdwba(weights,biases,activations,activations0)
%dzdwba calculates dzdw, dzdb, and dzdaprev of each layer to help calculate the
%gradient.
%   weights is a cell array of weights for each layer
%   biases is a cell array of biases for each layer
%   activations is a cell array of activations for almost all layers
%   activations0 is an array of activations for the first layer
%   dzdw is a cell array of dzdw for each weight
%   dzdb is a cell array of dzdb for each bias (=1)
%   dzdaprev is a cell array of nxm matrixs of weights to be multiplied by array dcdz to
%   result in nx1 array of dcdaprev
dzdw = cell(1,numel(weights));
dzdb = cell(1,numel(biases));
dzdaprev = cell(1,numel(activations));
for layer = 1:numel(dzdw)
    dzdw{layer} = zeros(size(weights{layer}));   
    for layerNeuron = 1:length(weights{layer}(:,1))
        if layer > 1
            dzdw{layer}(layerNeuron,:) = activations{layer-1}';
        else
            dzdw{layer}(layerNeuron,:) = activations0';
        end
    end
  
    dzdb{layer} = ones(size(biases{layer}));
    dzdaprev{layer} = sum(weights{layer},2)';
end
end

