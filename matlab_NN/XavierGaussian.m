function [layerWeights] = XavierGaussian(fanIn,fanOut)
%XavierGaussian Outputs the randomized layer weights using the
%Xavier/Glorot weight initialization method with Gaussian distribuition
%   fanIn Number of neurons/activations of previos layer
%   fanOut Number of neurons/activations of current layer
%   layerWeights fanOutxfanIn matrix of weights for this layer

stdDev = sqrt(2/(fanIn+fanOut));
mean = 0;
layerWeights = stdDev*randn(fanOut,fanIn) + mean;
end

