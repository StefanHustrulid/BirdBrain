function [dcdw,dcdb] = Gradient(weights,biases,activations0,idealOutputs,activationFunctionType,outputType)
%Gradient Returns the gradient of Cost in respect to Weights and Biases.

activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);

%Initialize cells
dcdw = cell(1,numel(weights));
dcdb = cell(1,numel(biases));

%Calculates dcdz of last layer
dcdz = dCostdz(activations{end}, idealOutputs, activations{end-1}, weights{end}, biases{end}, outputType);

%calculate dcdw and dcdb of last layer seperately because dcdz is
%calculated differently.
%Have to rotate activations into a 3rd dimension to be able to use with the
%2D weight matrices
%sum along the extra dimesion together to get dcdw and dcdb
%activationsRotated(:,1,:) = activations{end-1}; 
dcdw{end} = mean(ones(size(weights{end})).*permute(activations{end-1},[3,1,2]).*permute(dcdz,[1,3,2]),3);
dcdb{end} = mean(ones(size(activations{end})).*dcdz,2);

for layer = numel(dcdw)-1:-1:1
    
    %dcdz = dzPlus1da .* dadz .* dcdzPlus1
    dzPlusOneda = sum(weights{layer+1},1)';
    
    %for components that use activations0
    if layer == 1
        dadz = dActivationFunctiondz(activations0,weights{layer},biases{layer},activationFunctionType);       
        dcdz = dadz.*permute(sum(dzPlusOneda.*permute(dcdz,[3,1,2]),2),[1,3,2]); 
        %dcdz = dzPlusOneda.*dadz.*dcdz;
        dcdw{layer} = mean(ones(size(weights{layer})).*permute(activations0,[3,1,2]).*permute(dcdz,[1,3,2]),3);
        %activationsRotated(:,1,:) = activations0;
    else
        dadz = dActivationFunctiondz(activations{layer-1},weights{layer},biases{layer},activationFunctionType);
        dcdz = dadz.*permute(sum(dzPlusOneda.*permute(dcdz,[3,1,2]),2),[1,3,2]);
        %dcdz = dzPlusOneda.*dadz.*dcdz;
        dcdw{layer} = mean(ones(size(weights{layer})).*permute(activations{layer-1},[3,1,2]).*permute(dcdz,[1,3,2]),3);
        %activationsRotated(:,1,:) = activations{layer-1};
    end
    %dcdz = dzPlusOneda.*dadz.*dcdz;
    
    %calculate dcdw and dcdb of layer the same way as before
    %dcdw{layer} = sum(ones(size(weights{layer})).*permute(activationsRotated,[2,1,3]).*permute(dcdz,[1,3,2]),3);
    dcdb{layer} = mean(ones(size(activations{layer})).*dcdz,2);
end
