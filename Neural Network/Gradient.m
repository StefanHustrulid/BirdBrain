function [totaldCostdWeights,totaldCostdBiases] = Gradient(weights,biases,activations0,idealOutputs,activationFunctionType,outputType)
%Gradient Returns the gradient of Cost in respect to Weights and Biases.

numberOfLayers = numel(weights) + 1;
% Initialize the cell arrays of total change in weights and
% balances and the total cost.
totaldCostdWeights = cell(1,numberOfLayers-1);
totaldCostdBiases = cell(1,numberOfLayers-1);

for layer = 1:NumberOfLayers-1
    totaldCostdWeights{layer} = zeros(size(weights{layer}));
    totaldCostdBiases{layer} = zeros(size(biases{layer}));
end
for constraint = 1:size(activations0(1,:)) %For each a0&y in batch calculate change in weights and balances
    activations = RunNeuralNetwork(activations0(:,constraint),weights,biases,activationFunctionType,outputType);
            
    %Initialize dcdw, dcdb, and dcda-1 cell arrays
    dCostdWeights = cell(1,numberOfLayers-1);           
    dCostdBiases = cell(1,numberOfLayers-1);            
    dCostdPreviousActivations = cell(1,numberOfLayers-2);            
    dCostdWeights{numberOfLayers-1} = zeros(size(weights{numberOfLayers-1}));           
    dCostdBiases{numberOfLayers-1} = zeros(size(biases{numberOfLayers-1}));           
    for layer = 1:NumberOfLayers-2                
        dCostdWeights{layer} = zeros(size(weights{layer}));                
        dCostdBiases{layer} = zeros(size(biases{layer}));         
        dCostdPreviousActivations{layer} = zeros(size(activations{layer}));          
    end
    
    % Cell arrays of dzdw,dzdb,dzdaprev for each layer          
    [dzdw,dzdb,dzdaprev] = dzdwba(weights,biases,activations,activations0(:,constraint));            
    
    % Cell array of dadz for each hidden layer
    dadz = cell(1:numel(activations)-1);   
    dadz{1} = dActivationFunctiondz(activations0,weights{1},biases{1});
    if numel(activations)-1 > 1
        for layer = 2:numel(activations)-1
            dadz{layer} = dActivationsdz(activations{layer-1},weights{layer},biases{layer});
        end
    end
    
    % Calculate dcdw, dcdb, and dcda-1 of last layer (only last          
    % layer uses dcdz).
    dcdz = dCostdz(activations{numberOfLayers-1},idealOutputs(:,constraint),activations{numberOfLayers-2},weights{numberOfLayers-1},biases{numberOfLayers-1},outputType);
    dCostdWeights{numberOfLayers-1} = dzdw{numberOfLayers-1}.*dcdz;
    dCostdBiases{numberOfLayers-1} = dzdb{numberOfLayers-1}.*dcdz;
    dCostdPreviousActivations{numberOfLayers-2} = dzdaprev{numberOfLayers-2}*dcdz;
            
    %Add dcdw and dcdb of layer to corresponding total dcdw and
    %dcdb
    totaldCostdWeights{numberOfLayers-1} = totaldCostdWeights{numberOfLayers-1} + dCostdWeights{numberOfLayers-1};
    totaldCostdBiases{numberOfLayers-1} = totaldCostdBiases{numberOfLayers-1} + dCostdBiases{numberOfLayers-1};
            
    for layer = numberOfLayers-2:-1:1
        %calculate dcdw, dcdb, and dcdaprev for rest of layers
        dCostdWeights{layer} = dzdw{layer}.*dadz{layer}.*dCostdPreviousActivations{layer};               
        dCostdBiases{layer} = dzdb{layer}.*dadz{layer}.*dCostdPreviousActivations{layer};             
        dCostdPreviousActivations{layer} = dzdaprev{layer}*(dadz{layer}.*(dCostdPreviousActivations{layer}));
        %calculate total dcdw & dcdb
        totaldCostdWeights{layer} = totaldCostdWeights{layer} + dCostdWeights{layer};
        totaldCostdBiases{layer} = totaldCostdBiases{layer} + dCostdBiases{layer};
    end   
end
end

