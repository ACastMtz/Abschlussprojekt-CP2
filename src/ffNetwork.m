%========================================================================%
%                                                                        %
%  This function checks if the input parameters define a valid network   %
%  and stores the in a structure.                                        %
%                                                                        %
%  INPUT:                                                                %
%  - activationFunction : Object of the class "activation", where the    %
%                         function to compute each neuron's activation   %
%                         and its derivative are stored.                 %
%                         Default: sigmoid function.                     %
%  - eta :                Learning rate of the network.                  %
%  - batchSize:           Size of batches.                               %
%  - input:               Number of neurons on the input layer.          %
%  - output:              Number of neurons on the output layer.         %
%  - varargin:            Variable-length input argument list defining   %
%                         number of hidden layers and neurons in them.   %
%                                                                        %
%  OUTPUT:                                                               %
%  - model :              Structure with all the network's information.  %
%                                                                        %
%========================================================================%

function [model] = ffNetwork(activationFunction, eta, batchSize, input, output, varargin)
  % Input testing to prevent false input
  assert(nargin >=6,...
  'To define a ffnetwork you need at least 6 input arguments (positive integers).');

  assert(input==floor(input) && input > 0 && output==floor(output) && output > 0,...
  "Input arguments have to be positiv, non zero integers");
  len = size(varargin);
  for ii = 1:len(2)
    assert(cell2mat(varargin(1,ii))==floor(cell2mat(varargin(1,ii))) && ...
    cell2mat(varargin(1,ii)) > 0, "Input arguments have to be positiv, non zero integers");
  end
  assert(eta > 0, 'To learn properly eta has to be greater than zero');
  assert(batchSize==floor(batchSize) && batchSize > 0, 'batchSize has to be a positiv int');

  % Building the structure: storing all parameters defining the network
  model.input = input;
  model.output = output;
  model.hidden = cell2mat(varargin);
  model.eta = eta;
  model.batchSize = batchSize;
  model.activate = activationFunction.activate;
  model.deriveActivate = activationFunction.deriveActivation;
  model.averageOverEpochs = 5;
  model.averageCostChange = Inf;
end

