function [layerWeights] = LeCunGaussian(fanIn,fanOut)
%LeCunGaussian Outputs the randomized layer weights using the
%LeCun weight initialization method with Gaussian distribuition
%   fanIn Number of neurons/activations of previos layer
%   fanOut Number of neurons/activations of current layer
%   layerWeights fanOutxfanIn matrix of weights for this layer

stdDev = sqrt(1/fanIn);
mean = 0;
layerWeights = stdDev*randn(fanOut,fanIn) + mean;
end