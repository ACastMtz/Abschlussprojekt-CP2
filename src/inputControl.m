%========================================================================%
%                                                                        %
% Script including all functions needed to run a feedforward neural      % 
% network to identify handwritten numbers.                               %
% The MNIST data is loaded and divided in two sets: training and testing.%
% Aditionally, it provides an option to subdivide the data sets if the   %
% calculations need to be expedited.                                     %
% After defining the specifications and hyperparameters the network is   %
% initialized, trained and tested.                                       %
%                                                                        %
%========================================================================%

% LOADING MNIST DATA
testData = loadMNISTImages('t10k-images-idx3-ubyte');
testLabel = loadMNISTLabels('t10k-labels-idx1-ubyte');
trainData = loadMNISTImages('train-images-idx3-ubyte');
trainLabel= loadMNISTLabels('train-labels-idx1-ubyte');

% SUBDIVISION OF DATA SETS
sizeTest = 10000;       % Default 10000 (whole testing-set)
sizeTrain = 60000;      % Default 60000 (whole training-set)
testData = transpose(testData(:,1:sizeTest));
testLabel = testLabel(1:sizeTest);
trainData = transpose(trainData(:,1:sizeTrain));
trainLabel = trainLabel(1:sizeTrain);

% NETWORK'S PREREQUISITES AND HYPERPARAMETERS SET-UP
mapResults = 0:9;   % Maps the output nodes to the test labels
eta = 3.0;          % Learning rate. Default 3.0
batchSize = 10;    % Batch size for the stochastic gradient descent. Default 10
activationFunction = activation(@sig,@sigDerivativ);  % Default sigmoid function

% CONFIGURING THE NETWORK
% THE LAST INPUTS DEFINE THE NETWORK'S CONFIGURATION
% DEFAULT: 3 LAYERS TOTAL, 1 HIDDEN LAYER WITH 10 NEURONS
model = ffNetwork(activationFunction,eta,batchSize,784,10,30);

% TRAINING
fprintf ('Training network...\n');
tic
rng(1234);            % Seed for reproducibility
[modelParams, cost, epoch] = trainNetwork(model, trainData, trainLabel);
toc 
figure(1);
plot(1:epoch,cost)
xlabel('Epochs');
ylabel('Cost');
title('Cost Function on training-set');

% TESTING
fprintf ('Testing network...\n');
tic
modelResult = testNetwork(modelParams, testData, testLabel, mapResults);
toc
fprintf('Correct classified data: %0.2f percent \n', round(modelResult*100,2));
