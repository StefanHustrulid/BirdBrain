%Neural Network main commands

%% Initialization

numberOfHiddenLayers = input("Number of Hidden Layers: ");
numberOfNeuronsInHiddenLayer = zeros(1,numberOfHiddenLayers);
numberOfInputs = input("Number of Inputs: ");
numberOfOutputs = input("Number of Outputs: ");
numberOfLayers = numberOfHiddenLayers+2;
activations0 = zeros(numberOfInputs,1);
for hiddenLayer = 1:numberOfHiddenLayers
    numberOfNeuronsInHiddenLayer(hiddenLayer) = input(['Number of Neurons in Hidden Layer ' num2str(hiddenLayer) '/' num2str(numberOfHiddenLayers) ': ']);
end
numberOfNeuronsInLayer = [numberOfInputs, numberOfNeuronsInHiddenLayer, numberOfOutputs];
[activationFunctionType, ifAnswered] = listdlg('PromptString','What activation function are you using?','SelectionMode','single','ListString',{'Linear','Sigmoid','Tanh','ReLU','Leaky ReLU'});
[outputType, ifAnswered] = listdlg('PromptString','What is your output range?','SelectionMode','single','ListString',{'0%-100%','-1-1','-inf-inf','0-1','0-inf','-1-0','-inf-0'});
 
%Initial Weights and Biases

[newOrOld, ifAnswered] = listdlg('PromptString','Do you want randomized initial Weights & Balances or do you have them predetermined?','SelectionMode','single','ListString',{'Randomize','Input Predetermined'});

if newOrOld == 1
    randomWeightRange = [-5,5];
    randomBiasRange = [-5,5];
    weights = cell(1,numberOfLayers-1);
    biases = cell(1,numberOfLayers-1);
    for layer = 1:numberOfLayers-1
        weights{layer} = randomWeightRange(1)+((randomWeightRange(2)-randomWeightRange(1))*rand(numberOfNeuronsInLayer(layer+1),numberOfNeuronsInLayer(layer)));
        biases{layer} = randomBiasRange(1)+((randomBiasRange(2)-randomBiasRange(1))*rand(numberOfNeuronsInLayer(layer+1),1));
    end
else
    %Some Function that imports the weights and balances from somewhere
    %else
end

%% Use, Test, or Train?

[useTestOrTrain, ifAnswered] = listdlg('PromptString','Do you want to Use, Test, or Train your Neural Network?','SelectionMode','single','ListString',{'Use','Test','Train'});

if useTestOrTrain == 1 %Use: Input results in Output. No Ideal Output to Calculate cost with
    for i = 1:numberOfInputs %Temp until better way to import data
        activations0(i) = input(['What is input ' num2str(i) '/' num2str(numberOfInputs) '? ']);
    end
    activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);
    disp('Outputs = ');
    disp(activations{end});
elseif useTestOrTrain == 2 %Test: Input results in Output. Given Ideal Output to calculate Cost with.
    for i = 1:numberOfInputs %Temp until better way to import data
        activations0(i) = input(['Input ' num2str(i) '/' num2str(numberOfInputs) ': ']);
    end
    if outputType == 1 %Temp until better way to import data
        idealCatagory = input('The Output That Is True: ');
        idealOutputs = zeros(numberOfOutputs, 1);
        idealOutputs(idealCatagory) = 1;
    else
        for i = 1:numberOfOutputs %Temp until better way to import data
            idealOutputs = input(['Ideal Output ' num2str(i) '/' num2str(numberOfOutputs) ': ']);
        end
    end
    activations = RunNeuralNetwork(activations0,weights,biases,activationFunctionType,outputType);
    
    [costs,totalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
    disp('Outputs = ');
    disp(activations{numberOfLayers-1});
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
    %temp input method
    [activations0, idealOutputs] = input(['input ' num2str(numberOfInputs) 'xn matrix of inputs and ' num2str(numberOfOutputs) 'xn matrix of outputs where n = number of training data']);
    [trainingMethod, ifAnswered] = listdlg('PromptString','What is your Training Method?','SelectionMode','single','ListString',{'Gradient Descent','Stochastic Gradient Descent','GD With Momentum', 'SGD with Momentum'});
    if trainingMethod == 1
        goal = input('Max acceptable cost: ');
        maxNumberOfSteps = input('Max number of rounds of training: ');
        costs = zeros(numberOfOutputs,maxNumberOfSteps);
        totalCost = zeros(1,maxNumberofSteps);
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        [weights,biases] = GradientStep(weights,biases,activations0,idealOutputs,learningRate,activationFunctionType,outputType);
        while totalCost(step) >= goal || step < maxNumberOfSteps
            step = step + 1;
            [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
            [weights,biases] = GradientStep(weights,biases,activations0,idealOutputs,learningRate,activationFunctionType,outputType);
        end
        [finalCosts,finalTotalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        costs = [costs(:,1:step),finalCosts];
        totalCost = [totalCost(1:step),finalTotalCosts];
        
        plot(0:step,totalCost);
    elseif trainingMethod == 2
        goal = input('Max acceptable cost: ');
        maxNumberOfSteps = input('Max number of rounds of training: ');
        costs = zeros(numberOfOutputs,maxNumberOfSteps);
        totalCost = zeros(1,maxNumberofSteps);
        batchSize = input('Batch size: ');
        
        totalCost(1) = 100;
        step = 0;
        while totalCost(step) >= goal || step < maxNumberOfSteps
            batch = GenerateBatches(activations0,idealOutputs,batchSize);
            for round = 1:numel(batch)/2
                step = step + 1;
                [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
                [weights,biases] = GradientStep(weights,biases,batch(round,1),batch(round,2),learningRate,activationFunctionType,outputType); 
            end
        end
        [finalCosts,finalTotalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        costs = [costs(:,1:step),finalCosts];
        totalCost = [totalCost(1:step),finalTotalCosts];
        
        plot(0:length(batch(:,1)),totalCost);
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
        
        costs = zeros(numberOfOutputs,maxNumberOfSteps);
        totalCost = zeros(1,maxNumberofSteps);
        step = 1;
        [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,activations0,idealOutputs,totaldCostdWeights,totaldCostdBiases,learningRate,decayRate,activationFunctionType,outputType);
        while totalCost(step) >= goal || step < maxNumberOfSteps
            step = step + 1;
            [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
            [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,activations0,idealOutputs,totaldCostdWeights,totaldCostdBiases,learningRate,decayRate,activationFunctionType,outputType);
        end
        [finalCosts,finalTotalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        costs = [costs(:,1:step),finalCosts];
        totalCost = [totalCost(1:step),finalTotalCosts];
        
        plot(0:step,totalCost);
    elseif trainingMethod == 4
        goal = input('Max acceptable cost: ');
        maxNumberOfSteps = input('Max number of rounds of training: ');
        costs = zeros(numberOfOutputs,maxNumberOfSteps);
        totalCost = zeros(1,maxNumberofSteps);
        batchSize = input('Batch size: ');
        
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
        
        totalCost(1) = 100;
        step = 0;
        while totalCost(step) >= goal || step < maxNumberOfSteps
            batch = GenerateBatches(activations0,idealOutputs,batchSize);
            for round = 1:numel(batch)/2
                step = step + 1;
                [costs(step),totalCost(step)] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
                [weights,biases,totaldCostdWeights,totaldCostdBiases] = GradientStepWithMomentum(weights,biases,batch(round,1),batch(round,2),totaldCostdWeights,totaldCostdBiases,learningRate,decayRate,activationFunctionType,outputType); 
            end
        end
        [finalCosts,finalTotalCost] = TestNeuralNetwork(weights,biases,activations0,idealOutputs,activationFunctionType,outputType);
        costs = [costs(:,1:step),finalCosts];
        totalCost = [totalCost(1:step),finalTotalCosts];
        
        plot(0:length(batch(:,1)),totalCost);
    end
end

