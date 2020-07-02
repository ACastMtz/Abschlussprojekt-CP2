%========================================================================%
%                                                                        %
% Function where the output of the network is computed by applying       %
% the feedforward algorithm. In this particular case the neural          %
% network is used to classify handwritten numbers, so the output is      %
% the identified digit.                                                  %
%                                                                        %
% INPUT:                                                                 %
%  - modelParams: Structure with weights, biases and information of the  %
%                 trained network (model).                               %
%  - data:        Vector of size (1 X length of input), where the input  %
%                 contains the grayscale value for 784 pixels that make  %
%                 up an image, which is congruent with the labels in     %
%                 trainLabel.                                            %
%  - dataLabel:   Single entry containing the label (number's image) for %
%                 the input.                                             %
%  - mapResults:  Maps the output nodes to the label.                    %
%                                                                        %
% OUTPUT:                                                                %
% - unactivatedOutput: Cell array with matrixes of weghted inputs for    %
%                      the layers of the network.                        %
% - activatedOutput:   Cell array with column vectors of activation      %
%                      values for the network.                           %
% - classifiedCorrect: Single entry equal to 1 if input was classified   %
%                      correctly and 0 if not.                           %  
%                                                                        %
%========================================================================%

function [classifiedCorrect, unactivatedOutput, activatedOutput] = ...
                     evaluate(modelParams, data, dataLabel, mapResults)

  classifiedCorrect = 0; % default is false classified
  layers = length(modelParams.model.hidden) + 2;% number of layers

  % ACTIVATIONS
  activatedOutput = cell(layers ,1); 
  % WEIGHTED INPUTS
  unactivatedOutput = cell(layers-1,1);

 activatedOutput{1} = data(1,:);
 
 % PROCEEDING ITERATIVELY AND STORING ALL VALUES
 for jj = 1:layers-1
  % WEIGHTED INPUTS
    unactivatedOutput{jj} = activatedOutput{jj}...
                           * modelParams.weights{jj} - modelParams.biases{jj};
  % ACTIVATION VALUES
    activatedOutput{jj+1} = ...
                      modelParams.model.activate(unactivatedOutput{jj});
 end
    
 % CONTROL
 %% Check if the classification was correct
 [~, loc] = max(activatedOutput{layers});
 if exist('mapResults','var') && exist('dataLabel','var')
  if mapResults(loc) == dataLabel
    classifiedCorrect = 1;
  end
 end
end

