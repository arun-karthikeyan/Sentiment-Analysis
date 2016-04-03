%#% Project Opinion Mining - Author Arun Karthikeyan %#%

% This functions predicts the classfication of stocks according to the given input parameters %
% Prediction is performed using the sigmoid activation function %
% Theta1 - Parameters to compute the hidden layer %
% Theta2 - Parameters to compute output %
% X - Actual input to the network %

function predictions = NeuralNetworkD2Prediction(Theta1, Theta2, Theta3, X)

% m - no of. examples to predict %
m = size(X,1);

% Implementation of the feed forward algorithm to predict output %
a2 = sigmoid([ones(m,1) X] * Theta1');

a3 = sigmoid([ones(m,1) a2] * Theta2');

a4 = sigmoid([ones(m,1) a3] * Theta3');

% Final predictions based on a threshold of 0.5 %
predictions = round(a4);

end
