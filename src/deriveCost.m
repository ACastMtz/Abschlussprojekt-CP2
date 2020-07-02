function dC = deriveCost(desiredOutput,actualOutput)
%derivative of the Costfunction using quadratic mean
dC = transpose(actualOutput(:) - desiredOutput(:));
end

