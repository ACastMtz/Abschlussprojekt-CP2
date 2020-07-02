%========================================================================%
%                                                                        %
% This script uses the costOverEpochs()-function to plot the cost        %
% evaluated on the test dataset against the epochs for different         %
% parameter bundles. The parameters being changed include the number of  %
% training inputs (sizeTrain), the learning rate (eta), the batch size   %
% of the stochastic gradient descent (batchSize) and the number of       %
% neurons in the hidden layer (numNeurons). Only one network parameter   %
% is changed at a time while the others (variables with the suffix       %
% 'Usual') are kept constant. In addition, for the different numbers of  %
% training inputs and numbers of neurons, the computation time needed    %
% to train the network and evaluate the cost function is plotted.        %
%                                                                        %
%========================================================================%

epochs = 20;

% constant parameters
sizeTrainUsual = 60000;
numNeuronsUsual = 30;
etaUsual = 3;
batchSizeUsual = 10;

% variable parameters
sizeTrain = [5000, 10000, 30000, 60000];
eta = [0.6 1 3 5 10];
batchSize = [2 5 10 20 50 100 250];
numNeurons = [15 20 30 35 40 50 80 100];

totalTimer = tic; % for total time

%% change the number of training inputs
cost.sizeTrain = zeros(epochs, numel(sizeTrain));
times.sizeTrain = zeros(1,numel(sizeTrain));
legendTitles = cell(numel(sizeTrain), 1);
for k = 1:numel(sizeTrain)
    tic
    fprintf('varying sizeTrain = %d\n', sizeTrain(k)); 
    cost.sizeTrain(:,k) = costOverEpochs(sizeTrain(k),etaUsual,batchSizeUsual, numNeuronsUsual, epochs);
    legendTitles{k} = sprintf('sizeTrain = %d', sizeTrain(k));
    times.sizeTrain(k) = toc;
end
figure(1)
plot(1:epochs, cost.sizeTrain);
legend(legendTitles);
xlabel('epochs');
ylabel('cost');
saveas(gcf,'img/20epochs/costOverEpochs_sizeTrain.png');
saveas(gcf,'img/20epochs/costOverEpochs_sizeTrain', 'epsc');
figure(2)
plot(sizeTrain, times.sizeTrain);
xlabel('size of training dataset');
ylabel('calculation time (s)');
saveas(gcf,'img/20epochs/calctimeOverEpochs_sizeTrain.png');
saveas(gcf,'img/20epochs/calctimeOverEpochs_sizeTrain', 'epsc');

%% change learning rate
cost.eta = zeros(epochs, numel(eta));
legendTitles = cell(numel(eta), 1);
for k = 1:numel(eta)
    fprintf('varying eta = %d\n', eta(k));
    cost.eta(:,k) = costOverEpochs(sizeTrainUsual,eta(k),batchSizeUsual, numNeuronsUsual, epochs);
    legendTitles{k} = sprintf('eta = %d', eta(k));
end
figure(1)
plot(1:epochs, cost.eta);
legend(legendTitles);
xlabel('epochs');
ylabel('cost');
saveas(gcf,'img/20epochs/costOverEpochs_eta.png');
saveas(gcf,'img/20epochs/costOverEpochs_eta', 'epsc');

%% change batch size
cost.batchSize = zeros(epochs, numel(batchSize));
legendTitles = cell(numel(batchSize), 1);
for k = 1:numel(batchSize)
    fprintf('varying batchSize = %d\n', batchSize(k));
    cost.batchSize(:,k) = costOverEpochs(sizeTrainUsual,etaUsual,batchSize(k), numNeuronsUsual, epochs);
    legendTitles{k} = sprintf('batchSize = %d', batchSize(k));
end
figure(1)
plot(1:epochs, cost.batchSize);
legend(legendTitles);
xlabel('epochs');
ylabel('cost');
saveas(gcf,'img/20epochs/costOverEpochs_batchSize.png');
saveas(gcf,'img/20epochs/costOverEpochs_batchSize', 'epsc');

%% change number of neurons in the hidden layer
cost.numNeurons = zeros(epochs, numel(numNeurons));
times.numNeurons = zeros(1,numel(numNeurons));
legendTitles = cell(numel(numNeurons), 1);
for k = 1:numel(numNeurons)
    tic
    fprintf('varying numNeurons = %d\n', numNeurons(k));
    cost.numNeurons(:,k) = costOverEpochs(sizeTrainUsual,etaUsual,batchSizeUsual, numNeurons(k), epochs);
    legendTitles{k} = sprintf('%d hidden neurons', numNeurons(k));
    times.numNeurons(k) = toc;
end
figure(1)
plot(1:epochs, cost.numNeurons);
legend(legendTitles);
xlabel('epochs');
ylabel('cost');
saveas(gcf,'img/20epochs/costOverEpochs_numNeurons.png');
saveas(gcf,'img/20epochs/costOverEpochs_numNeurons', 'epsc');
figure(2)
plot(numNeurons, times.numNeurons);
xlabel('number of neurons (hidden layer)');
ylabel('calculation time (s)');
saveas(gcf,'img/20epochs/calctimeOverEpochs_numNeurons.png');
saveas(gcf,'img/20epochs/calctimeOverEpochs_numNeurons', 'epsc');

totaltime = toc(totalTimer)