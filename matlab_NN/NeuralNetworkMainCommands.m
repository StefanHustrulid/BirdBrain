%Neural Network main commands

%% Initialization

numberOfHiddenLayers = input('Number of Hidden Layers: ');
numberOfNeuronsInHiddenLayer = zeros(1,numberOfHiddenLayers);
numberOfInputs = input("Number of Inputs: ");
numberOfOutputs = input("Number of Outputs: ");
numberOfLayers = numberOfHiddenLayers+2;
for hiddenLayer = 1:numberOfHiddenLayers
    numberOfNeuronsInHiddenLayer(hiddenLayer) = input(['Number of Neurons in Hidden Layer ' num2str(hiddenLayer) '/' num2str(numberOfHiddenLayers) ': ']);
end
numberOfNeuronsInLayer = [numberOfInputs, numberOfNeuronsInHiddenLayer, numberOfOutputs];
[activationFunctionType, ifAnswered] = listdlg('PromptString','What activation function are you using?','SelectionMode','single','ListString',{'Linear','Sigmoid','Tanh','ReLU','LeakyReLU','ELU','SELU','GELU'});
[outputType, ifAnswered] = listdlg('PromptString','What is your output range?','SelectionMode','single','ListString',{'0%-100%','-1-1','-inf-inf','0-1','0-inf','-1-0','-inf-0'});
 
%Initial Weights and Biases

[newOrOld, ifAnswered] = listdlg('PromptString','Do you want randomized initial Weights & Balances or do you have them predetermined?','SelectionMode','single','ListString',{'Randomize','Input Predetermined'});

if newOrOld == 1
    
    %Initialize weight and bias cell arrays
    weights = cell(1,numberOfLayers-1);
    biases = cell(1,numberOfLayers-1);
    
    %Determine which weight initialization types are recomended
    if (activationFunctionType >=1) && (activationFunctionType <= 3)
        weightInitializationType = 1;
    elseif (activationFunctionType >=4) && (activationFunctionType <= 5)
        weightInitializationType = 2;
    elseif (activationFunctionType >=6) && (activationFunctionType <= 8)
        weightInitializationType = 3;
    end
    
    %Input desired weight initialization method
    if weightInitializationType == 1
        [weightInitializationMethod, ifAnswered] = listdlg('PromptString','What is your desired weight initialization method?','SelectionMode','single','ListString',{'Xavier/Glorot Uniform (recomended)','Xavier/Glorot Gaussian (recomended)','He/Kaiming Uniform','He/Kaiming Gaussian','LeCun Uniform','LeCun Gaussian'});
    elseif weightInitializationType == 2
        [weightInitializationMethod, ifAnswered] = listdlg('PromptString','What is your desired weight initialization method?','SelectionMode','single','ListString',{'Xavier/Glorot Uniform','Xavier/Glorot Gaussian','He/Kaiming Uniform (recomended)','He/Kaiming Gaussian (recomended)','LeCun Uniform','LeCun Gaussian'});
    elseif weightInitializationType == 3
        [weightInitializationMethod, ifAnswered] = listdlg('PromptString','What is your desired weight initialization method?','SelectionMode','single','ListString',{'Xavier/Glorot Uniform','Xavier/Glorot Gaussian','He/Kaiming Uniform','He/Kaiming Gaussian','LeCun Uniform (recomended)','LeCun Gaussian (recomended)'});
    end
    
    
    for layer = 1:numberOfLayers-1
        
        %Initialize weights using selected method
        if weightInitializationMethod == 1
            weights{layer} = XavierUniform(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        elseif weightInitializationMethod == 2
            weights{layer} = XavierGaussian(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        elseif weightInitializationMethod == 3
            weights{layer} = HeUniform(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        elseif weightInitializationMethod == 4
            weights{layer} = HeGaussian(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        elseif weightInitializationMethod == 5
            weights{layer} = LeCunUniform(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        elseif weightInitializationMethod == 6
            weights{layer} = LeCunGaussian(numberOfNeuronsInLayer(layer),numberOfNeuronsInLayer(layer+1));
        end
        
        %Initialize all biases to zero
        biases{layer} = zeros(numberOfNeuronsInLayer(layer+1),1);
    end
else
    %Some Function that imports the weights and balances from somewhere
    %else
end

%% Use, Test, or Train?

[useTestOrTrain, ifAnswered] = listdlg('PromptString','Do you want to Use, Test, or Train your Neural Network?','SelectionMode','single','ListString',{'Use','Test','Train'});

if useTestOrTrain == 1 %Use: Input results in Output. No Ideal Output to Calculate cost with
    %for i = 1:numberOfInputs %Temp until better way to import data
    %    activations0(i) = input(['What is input ' num2str(i) '/' num2str(numberOfInputs) '? ']);
    %end
    inputFileName = input('inputs file name (with .txt): ','s');
    activations0 = importdata(inputFileName);
    
    
    activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);
    disp('Outputs = ');
    disp(activations{end});
elseif useTestOrTrain == 2 %Test: Input results in Output. Given Ideal Output to calculate Cost with.
    
    
    %for i = 1:numberOfInputs %Temp until better way to import data
    %    activations0(i) = input(['Input ' num2str(i) '/' num2str(numberOfInputs) ': ']);
    %end
    %if outputType == 1 %Temp until better way to import data
    %    idealCatagory = input('The Output That Is True: ');
    %    idealOutputs = zeros(numberOfOutputs, 1);
    %    idealOutputs(idealCatagory) = 1;
    %else
    %    for i = 1:numberOfOutputs %Temp until better way to import data
    %        idealOutputs = input(['Ideal Output ' num2str(i) '/' num2str(numberOfOutputs) ': ']);
    %    end
    %end
    
    inputFileName = input('inputs file name (with .txt): ','s');
    activations0 = importdata(inputFileName);
    idealOutputsFileName = input('ideal outputs file name (with .txt): ','s');
    idealOutputs = importdata(idealOutputsFileName);
    
    %modify to handle matrix input
    %activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);
    
    [costs,totalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
    %disp('Outputs = ');
    %disp(activations{end});
    if outputType ~= 1
        disp('Output Costs = ');
        disp(costs);
    end
    disp('Total Cost = ');
    disp(totalCost);
else %Train Given inputs with ideal outputs to train the neural network with
    learningRate = 0.000001;
    decayRate = 0.000001;
    [changeLearningRate, ifAnswered] = listdlg('PromptString',['Chang learning rate from ' num2str(learningRate) '?'],'SelectionMode','single','ListString',{'Yes','No'});
    if changeLearningRate == 1
        learningRate = input('Learning Rate: ');
    end
    
    %inputFileName = input('inputs file name (with .txt): ','s');
    inputFileName = 'inputs.txt';
    activations0 = importdata(inputFileName);
    %idealOutputsFileName = input('ideal outputs file name (with .txt): ','s');
    idealOutputsFileName = 'idealOutputs.txt';
    idealOutputs = importdata(idealOutputsFileName);
    
    [trainingMethod, ifAnswered] = listdlg('PromptString','What is your Training Method?','SelectionMode','single','ListString',{'Gradient Descent','Stochastic Gradient Descent','GD With Momentum', 'SGD with Momentum'});
    
    if trainingMethod == 1
        goal = input('Max acceptable cost: ');
        maxNumberOfSteps = input('Max number of rounds of training: ');
        costs = zeros(numberOfOutputs,maxNumberOfSteps+1);
        totalCost = zeros(1,maxNumberofSteps+1);
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        while totalCost(step) >= goal || step <= maxNumberOfSteps
            [weights,biases] = GradientStep(weights,biases,activations0,idealOutputs,learningRate,activationFunctionType,outputType);
            step = step + 1;
            [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        end
        costs = costs(:,1:step);
        totalCost = totalCost(1:step);
        
        plot(0:step,totalCost);
    elseif trainingMethod == 2
        goal = input('Max acceptable cost: ');
        maxNumberOfEpochs = input('Max number of epochs: ');
        batchSize = input('Batch size: ');
        tempBatch = GenerateBatches(activations0,idealOutputs,batchSize);
        stepsPerEpoch = numel(tempBatch);
        maxNumberOfSteps = maxNumberOfEpochs*stepsPerEpoch;
        
        costs = zeros(numberOfOutputs,maxNumberOfSteps+1);
        totalCost = zeros(1,maxNumberOfSteps+1);
        
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        while totalCost(step) >= goal || step <= maxNumberOfSteps
            batch = GenerateBatches(activations0,idealOutputs,batchSize);
            for round = 1:numel(batch)/2
                [weights,biases] = GradientStep(weights,biases,batch{round,1},batch{round,2},learningRate,activationFunctionType,outputType); 
                step = step + 1;
                [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
                if totalCost(step) <= goal
                    break;
                end
            end
        end
        costs = costs(:,1:step);
        totalCost = totalCost(1:step);
        epochAxis = (1:length(totalCost))./stepsPerEpoch;
        plot(epochAxis,totalCost);
    elseif trainingMethod == 3
        goal = input('Max acceptable cost: ');
        maxNumberOfSteps = input('Max number of rounds of training: ');
        
        [changeDecayRate, ifAnswered] = listdlg('PromptString',['Change decay rate from ' num2str(decayRate) '?'],'SelectionMode','single','ListString',{'Yes','No'});
        if changeDecayRate == 1
            decayRate = input('Decay Rate: ');
        end
        totaldCostdWeights = cell(1,numberOfLayers-1);
        totaldCostdBiases = cell(1,numberOfLayers-1);
        for layer = 1:NumberOfLayers-1
            totaldCostdWeights{layer} = zeros(size(weights{layer}));
            totaldCostdBiases{layer} = zeros(size(biases{layer}));
        end
        
        costs = zeros(numberOfOutputs,maxNumberOfSteps+1);
        totalCost = zeros(1,maxNumberofSteps+1);
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        while totalCost(step) >= goal || step <= maxNumberOfSteps
            [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,activations0,idealOutputs,totaldCostdWeights,totaldCostdBiases,learningRate,decayRate,activationFunctionType,outputType);
            step = step + 1;
            [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        end
        costs = costs(:,1:step);
        totalCost = totalCost(1:step);
        
        plot(0:step,totalCost);
    elseif trainingMethod == 4
        goal = input('Max acceptable cost: ');
        maxNumberOfEpochs = input('Max number of epochs: ');
        batchSize = input('Batch size: ');
        stepsPerEpoch = numel(GenerateBatches(activations0,idealOutputs,batchSize));
        maxNumberOfSteps = maxNumberOfEpochs*stepsPerEpoch;
        
        costs = zeros(numberOfOutputs,maxNumberOfSteps+1);
        totalCost = zeros(1,maxNumberofSteps+1);
        
        [changeDecayRate, ifAnswered] = listdlg('PromptString',['Change decay rate from ' num2str(decayRate) '?'],'SelectionMode','single','ListString',{'Yes','No'});
        if changeDecayRate == 1
            decayRate = input('Decay Rate: ');
        end
        totaldCostdWeights = cell(1,numberOfLayers-1);
        totaldCostdBiases = cell(1,numberOfLayers-1);
        for layer = 1:NumberOfLayers-1
            totaldCostdWeights{layer} = zeros(size(weights{layer}));
            totaldCostdBiases{layer} = zeros(size(biases{layer}));
        end
        
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        while totalCost(step) >= goal || step <= maxNumberOfSteps
            batch = GenerateBatches(activations0,idealOutputs,batchSize);
            for round = 1:numel(batch)/2
                [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,batch{round,1},batch{round,2},totaldCostdWeights,totaldCostdBiases,learningRate,decayRate,activationFunctionType,outputType); 
                step = step + 1;
                [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
                if totalCost(step) <= goal
                    break;
                end
            end
        end
        costs = costs(:,1:step);
        totalCost = totalCost(1:step);
        
        epochAxis = (1:length(totalCost))./stepsPerEpoch;
        plot(epochAxis,totalCost);
    end
end

