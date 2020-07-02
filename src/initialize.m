%========================================================================%
%                                                                        %
%  This function initializes the network with normally distributed       %
%  weights and biases.                                                   %
%                                                                        %
%  INPUT:                                                                %
%  - model:       Structure where all the information of the network     %
%                 is stored.                                             %
%                                                                        %
%  OUTPUT:                                                               %
%  - weights:     Cell array with initial (random) weights.              %
%  - biases:      Cell array with initial (random) biases.               %
%                                                                        %
%========================================================================%

function [weights, biases] = initialize(model)
 len = length(model.hidden);
 
 % PREALLOCATING IN CELL ARRAYS
 weights = cell(len+1,1);
 biases = cell(len+1,1);
 
 weights{1} = randn(model.input, model.hidden(1));
 biases{1} = randn(1,model.hidden(1));
 if(len > 1)
     for ii = 2:len
        weights{ii} = randn(model.hidden(ii-1), model.hidden(ii));
        biases{ii} = randn(1,model.hidden(ii));
     end
 end
 weights{len+1} = randn(model.hidden(len), model.output);
 biases{len+1} = randn(1,model.output);
end

