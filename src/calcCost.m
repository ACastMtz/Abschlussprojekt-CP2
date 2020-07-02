%========================================================================%
%                                                                        %
%  This function computes the cost function for a set of inputs (data)   %
%  and corresponding expected outputs (labels).                          %                                            %
%                                                                        %
%  INPUT:                                                                %
%  - modelParams: Structure with weights, biases and information of the  %
%                 trained network (model).                               %
%  - data:        Matrix of size (number of inputs X length of input),   %
%                 where each input contains the grayscale value for      %
%                 784 pixels that make up an image, which is congruent   %
%                 with the labels in trainLabel.                         %
%  - labels:      Column vector of size (number of input X 1) containing %
%                 the labels (numbers' images) for each input.           %
%                                                                        %
%  OUTPUT:                                                               %
%  - cost :       Cost function for the set of inputs "data".            %
%                                                                        %
%========================================================================%

function cost = calcCost(modelParams, data, labels)
  
  % PREALLOCATING
  numSamples = size(data,1);
  costVec = zeros(1,numSamples);
  labelsVec = zeros(1,10);
  mapResults = 0:9;

  % EVALUATING COST FUNCTION FOR EACH INPUT
  for k = 1:numSamples
    % kth-LABEL IN VECTOR FORM
    labelsVec(labels(k)+1) = 1;
    
    % FEEDFORWARD TO OBATAIN OUTPUT 
    [~,~,activatedOutput] = evaluate(modelParams, data(k,:),...
                                     labels(k), mapResults);
    
    % CALCULATING COST OF THE kth-INPUT
    costVec(k) = norm(labelsVec-activatedOutput{end})^2;
    
    % RESTORING LABEL VECTOR
    labelsVec(labels(k)+1) = 0;
  end
  
  % TOTAL COST
  cost = 1/(2*numSamples) * sum(costVec);
end