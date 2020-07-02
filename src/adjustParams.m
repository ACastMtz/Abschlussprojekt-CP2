%========================================================================%
%                                                                        %
%  Function where the weights and biases are updated by substracting the %
%  next step of the gradient's descent (of the cost function) stemming   %
%  from the backpropagation calculations.                                %                                                         %
%                                                                        %
%  INPUT:                                                                %
%  - modelParams: Structure with weights, biases and information of the  %
%                 trained network (model).                               %
%  - trainData:   Matrix of size (number of inputs in a batch X length   %
%                 of input), where each input contains the grayscale     %
%                 value for 784 pixels that make up an image, which is   %
%                 congruent with the labels in trainLabel.               %
%  - trainLabel:  Column vector of size (number of input X 1) containing %
%                 the labels (numbers' images) for each input.           %
%                                                                        %
%  OUTPUT:                                                               %
%  - modelParams: Weights and biases of the trained network (after one   %
%                 iteration).                                            %
%                                                                        %
%========================================================================%


function modelParams = adjustParams(modelParams, trainData, trainLabel)
  batchSize = length(trainData(:,1));
  eta = modelParams.model.eta;

  % PROPAGATING BACKWARDS THROUGH EACH BATCH
  for ii = 1:batchSize
    % COMPUTING GRADIENT OF COST FUNCTION WITH RESPECT TO THE
    % WEIGHTS AND BIASES OF THE NETWORK WITH BACKPROPAGATION
    [deltaGradWeights, deltaGradBiases] = backprop(modelParams,...
                                           trainData(ii,:), trainLabel(ii,:));
    % UPDATING WEIGHTS AND BIASES
    for jj = 1:length(deltaGradWeights)
        modelParams.weights{jj} = ...
                modelParams.weights{jj}-(eta/batchSize).* deltaGradWeights{jj};
        modelParams.biases{jj} = ...
                modelParams.biases{jj}-(eta/batchSize).* deltaGradBiases{jj};
    end
    
  end
end

