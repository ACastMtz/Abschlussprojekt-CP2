%========================================================================%
%                                                                        %
% Function where the gradient for the cost function is computed by means %
% of the backpropagation algorithm.                                      %
%                                                                        %
% INPUT:                                                                 %
%  - modelParams: Structure with weights, biases and information of the  %
%                 trained network (model).                               %
%  - trainData:   Vector of size (1 X length of input), where the input  %
%                 contains the grayscale value for 784 pixels that make  %
%                 up an image, which is congruent with the labels in     %
%                 trainLabel.                                            %
%  - trainLabel:  Single entry containing the label (number's image) for %
%                 the input.                                             %
%                                                                        %
% OUTPUT:                                                                %
% - deltaGradWeights: Cell array with gradient of the cost function for  %
%                     all weights.                                       %
% - deltaGradBiases:  Cell array with gradient of the cost function for  %
%                     all biases.                                        %
%                                                                        %
%========================================================================%

function [deltaGradWeights, deltaGradBiases] = backprop(modelParams, ...
                                                  trainData, trainLabel)
                                                  
  % VECTORIZING THE LABELS OF THE INPUTS
  desiredOutput = zeros(modelParams.model.output,1);
  desiredOutput(trainLabel(1,1)+1) = 1;

  % COMPUTING THE WEIGHTED INPUTS AND ACTIVATIONS OF THE NETWORK
  %% Feedforward
    [~, unactivatedOutput, activatedOutput] = evaluate(modelParams, trainData);
  
  % NUMBER OF LAYERS
  layers = length(modelParams.model.hidden) + 2;

  % PREALLOCATION CELL ARRAYS FOR THE GRADIENT
  deltaGradWeights = cell(layers-1,1);
  deltaGradBiases = cell(layers-1,1);
  
  % BACKPROPAGATION
  deltaGradBiases{layers-1} = deriveCost(desiredOutput, activatedOutput{layers})...
                .*modelParams.model.deriveActivate(unactivatedOutput{layers-1});
  deltaGradWeights{layers-1} = transpose(activatedOutput{layers-1})...
                *deltaGradBiases{layers-1};
  for ii = 2:layers-1
      deltaGradBiases{layers-ii} = deltaGradBiases{layers-ii+1} ...
                * transpose(modelParams.weights{layers-ii+1}) ...
                .*modelParams.model.deriveActivate(unactivatedOutput{layers-ii});
      deltaGradWeights{layers-ii} = transpose(activatedOutput{layers-ii})...
                *deltaGradBiases{layers-ii};
  end
end

