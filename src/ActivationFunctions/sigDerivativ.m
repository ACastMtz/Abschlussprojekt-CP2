function result = sigDerivativ(x)
% sigDerivativ() calculates the derivitive of the sigmoid function of an input x
result = sig(x).*(1-sig(x));
end