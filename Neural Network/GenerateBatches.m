function batch = GenerateBatches(activations0,idealOutputs,batchSize)
%GenerateBatches Shuffles activations0 and idealOutputs then splits the
%data into batches
%   batch is an nx2 cell array where n is the total number of batches.
%   The first column is from activations0 and the second column is from
%   idealOutputs.
dataSize = length(activations0(1,:));
newOrder = randperm(dataSize);
activations0 = activations0(:,newOrder);
idealOutputs = idealOutputs(:,newOrder);

numberOfBatches = floor(dataSize/batchSize);
lastBatchSize = rem(dataSize/batchSize);      
if lastBatchSize == 0
    batch = cell(numberOfBatches,2);            
    batch{1,1} = activations0(:,1:batchSize);
    batch{1,2} = idealOutputs(:,1:batchSize);
    for b = 2:numberOfBatches
        batch{b,1} = activations0(:,(batchSize^(b-1))+1:batchSize^b);
    end    
else    
    batch = cell(numberOfBatches+1,2);
    batch{1,1} = activations0(:,1:batchSize);
    batch{1,2} = idealOutputs(:,1:batchSize);
    for b = 2:numberOfBatches       
        batch{b,1} = activations0(:,(batchSize^(b-1))+1:batchSize^b);
    end   
    b = numberOfBatches+1;
    batch{b,1} = activations0(:,(batchSize^(b-1))+1:dataSize);
end
end

