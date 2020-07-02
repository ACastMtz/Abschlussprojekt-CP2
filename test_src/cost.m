function C = cost(desiredOutput,actualOutput)
%Costfunction using quadratic mean
C = 0.5 * sum(abs(desiredOutput(:) - actualOutput(:)).^2);
end

