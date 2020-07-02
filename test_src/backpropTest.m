%========================================================================%
% This Skript is for testing the implementation of the function          %
% backprop.m which does the analytical calculation of the cost           %
% derivatives g(x).                                                      %
% Therefore this Skript calculates the same derivative using the formula:%
%  f'(x) ≈ (f(x + epsilon * v) - f(x - epsilon * v)) / 2epsilon      (1) %
%                                                                        %
% Finally it compares both derivatives and checks for:                   %
%                                                                        %
%  f'(x) - g(x) * v = O(epsilon)                                     (2) %
% It is chcked for each weight and bias.                                 %
%========================================================================%

rng(1234); % REPRODUCABILITY

% SET UP A SIMPLE NETWORK FOR TESTING PURPOSES
activationFunction = activation(@sig,@sigDerivativ);
eta = 5;
batchSize = 1;
model = ffNetwork(activationFunction,eta,batchSize,20,10,30);

% INITIALIZE RANDOM WEIGHTS AND BIASES
modelParams1.model = model;
[modelParams1.weights, modelParams1.biases] = initialize(model);
modelParamsV.model = model;
[modelParamsV.weights, modelParamsV.biases] = initialize(model);

modelParams2.model = model;
modelParams3.model = model;

epsilon = 10^-4; % WAS SUGGESTED IN SEVERAL RESOURCES
randomInput = randi([0 9],1,20);
randomLabel = randi([0 9],1,1);
mapResults = 0:9;

% CALCULATE THE GTADIENT WITH BACKPROP
[deltaGradWeights, deltaGradBiases] = backprop(modelParams1,randomInput, randomLabel);

% INTIALIZE PARAMETERS FOR NUMERICAL CALCULATION OF THE DERIVATIVE
for jj = 1:length(modelParams1.weights)
    modelParams2.weights{jj} = modelParams1.weights{jj};
    modelParams3.weights{jj} = modelParams1.weights{jj};
    modelParams2.biases{jj} = modelParams1.biases{jj};
    modelParams3.biases{jj} =  modelParams1.biases{jj};
end

% CHECK DERIVATIVES OF THE WEIGHTS
for ii = 1:length(modelParams2.weights)
    
    [kkMax,llMax] = size(modelParams2.weights{ii});
    for kk = 1:kkMax
        for ll = 1:llMax
            
            % ALWAYS RESET WEIGHTS
            for jj = 1:length(modelParams2.weights)
                modelParams2.weights{jj} = modelParams1.weights{jj};
                modelParams3.weights{jj} = modelParams1.weights{jj};
            end
              
            modelParams2.weights{ii}(kk,ll) = modelParamsV.weights{ii}(kk,ll).*epsilon + modelParams1.weights{ii}(kk,ll);
            modelParams3.weights{ii}(kk,ll) = -modelParamsV.weights{ii}(kk,ll).*epsilon + modelParams1.weights{ii}(kk,ll);
            
            % CALCULATE f(x)
            [~,~,activatedOutput2] = evaluate(modelParams2, randomInput, randomLabel, mapResults);
            [~,~,activatedOutput3] = evaluate(modelParams3, randomInput, randomLabel, mapResults);
            desiredOutput = zeros(modelParamsV.model.output,1);
            desiredOutput(randomLabel+1) = 1;
            f2= cost(desiredOutput, activatedOutput2{3});
            f3= cost(desiredOutput, activatedOutput3{3});
            
            % COMPARE DERIVATES ACCORDING EQUATION (1)
            analyticalDerivativ = deltaGradWeights{ii}(kk,ll) * modelParamsV.weights{ii}(kk,ll);
            numericalDerivativ = (f2-f3)/(2*epsilon);
            result = numericalDerivativ - abs(analyticalDerivativ); % SHOULD BE OF THE ORDER OF EPSILON OR LESS
            assert(result < epsilon, 'Attention wrong derivative!');
        end
    end
end

% CHECK DERIVATIVES OF THE BIASES
for ii = 1:length(modelParams2.weights)
    [kkMax,llMax] = size(modelParams2.weights{ii});
    for kk = 1:kkMax
        for ll = 1:llMax
            
            % ALWAYS RESET BIASES
            for jj = 1:length(modelParams2.weights)
                modelParams2.biases{jj} = modelParams1.biases{jj};
                modelParams3.biases{jj} =  modelParams1.biases{jj};
            end
            
            modelParams2.biases{ii}(1,1) = modelParamsV.biases{ii}(1,1).*epsilon + modelParams1.biases{ii}(1,1);
            modelParams3.biases{ii}(1,1) = -modelParamsV.biases{ii}(1,1).*epsilon + modelParams1.biases{ii}(1,1);
            
            % CALCULATE f(x)
            [~,~,activatedOutput2] = evaluate(modelParams2, randomInput, randomLabel, mapResults);
            [~,~,activatedOutput3] = evaluate(modelParams3, randomInput, randomLabel, mapResults);
            desiredOutput = zeros(modelParamsV.model.output,1);
            desiredOutput(randomLabel+1) = 1;
            f2= cost(desiredOutput, activatedOutput2{3});
            f3= cost(desiredOutput, activatedOutput3{3});
            
            % COMPARE DERIVATES ACCORDING EQUATION (1)
            analyticalDerivativ = deltaGradBiases{ii}(1,1) * modelParamsV.biases{ii}(1,1);
            numericalDerivativ = (f2-f3)/(2*epsilon) ;
            result = numericalDerivativ - abs(analyticalDerivativ); % SHOULD BE OF THE ORDER OF EPSILON OR LESS
            assert(result < epsilon, 'Attention wrong derivative!');
        end
    end
end
disp('Backpropagations works correctly!');
