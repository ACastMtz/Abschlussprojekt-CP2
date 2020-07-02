%========================================================================%
%                                                                        %
%  This function trains a neural network on a training data set using    %
%  the stochastic gradient descent to reduce the computational cost      %
%  when calculating the gradient of the cost function by means of the    %
%  backpropagation algorithm; i.e. the input data is divided in batches, %
%  the gradient descent is computed for each batch and the whole data    %
%  set is then randomly rearanged after each iteration (epoch) of the    %
%  training process.                                                     %
%                                                                        %
%  INPUT:                                                                %
%  - model:       Structure where all the information of the network     %
%                 is stored.                                             %
%  - trainData:   Matrix of size (number of inputs X length of input),   %
%                 where each input contains the grayscale value for      %
%                 784 pixels that make up an image, which is congruent   %
%                 with the labels in trainLabel.                         %
%  - trainLabel:  Column vector of size (number of input X 1) containing %
%                 the labels (numbers' images) for each input.           %
%                                                                        %
%  OUTPUT:                                                               %
%  - modelParams: Structure with weights, biases and information of the  %
%                 trained network (model).                               %
%  - cost:        Cost of each input of the training data set.           %
%  - epoch:       Number of total epochs (iterations) of the training,   %
%                 once the desired accuracy was achieved.                %
%                                                                        %
%========================================================================%

function [modelParams,cost,epoch] = trainNetwork(model, trainData, trainLabel)
  
  modelParams.model = model;
    
  % INITIALIZING NEURAL NETWORK
  [modelParams.weights, modelParams.biases] =  initialize(model);
  
  costChange = zeros(1,model.averageOverEpochs);
  epoch = 0;
  
  % TRAINING THE NETWORK WITH TWO STOPPING CRITERIA:
  % AVERAGE SLOPE OF COST FUNCTION < 10^-4 OR EPOCH > 50
  while (model.averageCostChange > 10^-4) && (epoch < 50)
    epoch = epoch + 1;
 
    % DIVIDING DATASET INTO BATCHES AND SHUFFLING INPUTS AND THEIR LABELS
    sizeTrain = length(trainData(:,1));
    randomIndex = randperm(sizeTrain);
    trainData = trainData(randomIndex,:); 
    trainLabel = trainLabel(randomIndex); 
    batchSize = modelParams.model.batchSize;
    
    % COMPUTING OPTIMAL WEIGHTS AND BIASES OF EACH BATCH 
    % USING GRADIENT DESCENT
    for ii = 1:sizeTrain/batchSize
        modelParams = adjustParams(modelParams,...
                    trainData((ii*batchSize-(batchSize-1)):(ii*batchSize),:),...
                    trainLabel((ii*batchSize-(batchSize-1)):(ii*batchSize)));
    end
    
    % COMPUTING COST FUNCION ON TRAINING-SET
    cost(epoch) = calcCost(modelParams, trainData, trainLabel);
    
    % AVERAGING CHANGE OF COST FUNCTION OVER THE LAST 5 ITERATIONS
    % TO COMPARE IT WITH THE STOPPING CRITERION
    if epoch>=5
        for k = 1:model.averageOverEpochs-1
            costChange(k) = cost(epoch-k+1)-cost(epoch-k);
        end
        model.averageCostChange = abs(sum(costChange)/(model.averageOverEpochs-1));
    end 
  
  end
  fprintf('Learning stopped after %d epochs.\n', epoch);
end