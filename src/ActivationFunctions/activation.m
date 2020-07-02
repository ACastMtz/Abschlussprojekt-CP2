classdef activation < handle
%% This is a class for managing the activation function
    properties
        activate; %stores the activationfunction
        deriveActivation; % stores the derivativ of the activationfunction
    end

    methods 

        function obj = activation(Function, Derivativ)
            obj.activate = Function;
            obj.deriveActivation = Derivativ;  
        end
    end    
end