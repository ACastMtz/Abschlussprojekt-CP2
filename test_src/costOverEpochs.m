%========================================================================%
%                                                                        %
% As the trainNetwork()-function this function trains a neural network   %
% with a given set of network parameters and one hidden layer for        %
% a given number of epochs. It then evaluates the cost function on the   %
% test dataset after each epoch. 		                                 %  
%                                                                        %
%  INPUT (all scalar):                                                   %
%  - sizeTrain:   Number of training inputs                              %
%  - eta:         Learning rate                                          %
%  - batchSize:   Number of inputs in each training batch                %
%  - numNeurons:  Number of neurons in the hidden layer                  %
%  - epochs:      Number of epochs                                       %
%                                                                        %
%  OUTPUT:                                                               %
%  - cost:        Vector of size (1 x number of epochs) containing the   %
%                 total cost evaluated on the test dataset for each      %
%                 epoch                                                  %
%                                                                        %
%========================================================================%

function [cost] = costOverEpochs(sizeTrain, eta, batchSize, numNeurons, epochs)

% Load MNIST data
testData = loadMNISTImages('t10k-images-idx3-ubyte');
testLabel = loadMNISTLabels('t10k-labels-idx1-ubyte');
trainData = loadMNISTImages('train-images-idx3-ubyte');
trainLabel = loadMNISTLabels('train-labels-idx1-ubyte');

testData = transpose(testData);
% narrow down the training set size
trainData = transpose(trainData(:,1:sizeTrain));
trainLabel = trainLabel(1:sizeTrain);

% specify network parameters
activationFunction = activation(@sig,@sigDerivativ);
model = ffNetwork(activationFunction,eta,batchSize,784,10,numNeurons);
modelParams.model = model;

% seed for reproducibility
rng(1234);

[modelParams.weights, modelParams.biases] =  initialize(model);

% loop over given number of epochs
for epoch = 1:epochs
    
    % divide the set into batches and adjust the parameters for each batch
    sizeTrain = length(trainData(:,1));
    randomIndex = randperm(sizeTrain);
    trainData = trainData(randomIndex,:); % shuffle the train set
    trainLabel = trainLabel(randomIndex); % shuffle the labels accordingly
    
    batchSize = modelParams.model.batchSize;
    for ii = 1:sizeTrain/batchSize
        % adjust the weights and biases for each batch
        modelParams = adjustParams(modelParams, trainData((ii*batchSize-(batchSize-1)):(ii*batchSize),:), trainLabel((ii*batchSize-(batchSize-1)):(ii*batchSize)));
    end
    
    % evaluate cost function on the test dataset for the current epoch
    cost(epoch) = calcCost(modelParams, testData, testLabel);
end

end
