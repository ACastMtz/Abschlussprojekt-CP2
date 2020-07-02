%========================================================================%
%                                                                        %
%  This function tests a trained network with a test-set (testData) and  %
%  its labels (testLabel) and returns the ratio of correctly             %
%  identified numbers.                                                   %
%                                                                        %
%  INPUT:                                                                %
%  - model:       Structure where all the information of the network     %
%                 is stored.                                             %
%  - testData:    Matrix of size (number of inputs X length of input),   %
%                 where each input contains the grayscale value for      %
%                 784 pixels that make up an image, which is congruent   %
%                 with the labels in trainLabel.                         %
%  - testLabel:   Column vector of size (number of input X 1) containing %
%                 the labels (numbers' images) for each input.           %
%                                                                        %
%  OUTPUT:                                                               %
%  - modelResult: Ratio of correctly identified inputs over total number %
%                 of inputs.                                             %
%                                                                        %
%========================================================================%

function modelResult = testNetwork(modelParams,testData, testLabel, mapResults)
  % PREALLOCATING
  results = zeros(length(testData(:,1)),1);
  
  % RUN EACH VECTOR OF THE TESTING-SET THROUGH THE TRAINED NETWORK
  for ii = 1:length(testData(:,1))
      % FEEDFORWARD
      [results(ii), ~, ~] = evaluate(modelParams, testData(ii,:),...
                             testLabel(ii,:),mapResults);
  end
  
  % COMPUTING RATIO
  modelResult = sum(results)/length(results);
end

