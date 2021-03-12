function [layerWeights] = HeUniform(fanIn,fanOut)
%HeUniform Outputs the randomized layer weights using the
%He/Kaiming weight initialization method with Uniform distribuition
%   fanIn Number of neurons/activations of previos layer
%   fanOut Number of neurons/activations of current layer
%   layerWeights fanOutxfanIn matrix of weights for this layer
var = sqrt((2*6)/fanIn);
mean = 0;
layerWeights = var*rand(fanOut,fanIn) + mean;
end