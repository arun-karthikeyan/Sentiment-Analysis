%#% Project Opinion Mining - Author Arun Karthikeyan %#%

% The function computes the cost and gradient of the network D2 %
% Make sure to check the correctness of the implementation with Numerical Gradient Checking %
% The implementation makes use of the sigmoid activation function %

% Parameters - %
% vectorParams - the unrolled version of the original parameters, for compatibility with advanced optimzation techniques (fmincg) %
% ipSize - size of the input layer w/o including bias %
% hlSize - no of activation units in the hiddenlayer %
% X - the input vector X of the original data %
% y - the output y of the original data %

function [J grad] = NeuralNetworkD2Cost(vectorParams, ipSize, hlSize1, hlSize2, X, y, lambda)

% Rearranging the parameters in vector params %

Theta1 = reshape(vectorParams(1:hlSize1 * (ipSize + 1)), hlSize1, (ipSize + 1));
Theta2 = reshape(vectorParams((1 + (hlSize1 * (ipSize + 1))):((hlSize1 * (ipSize + 1)) + (hlSize2 * (hlSize1 + 1)))), hlSize2, (hlSize1 + 1));
Theta3 = reshape(vectorParams((((hlSize1 * (ipSize + 1)) + (hlSize2 * (hlSize1 + 1)))+1) : end),1,hlSize2+1);

% m - No of training examples %
% n - No of features %
[m n] = size(X);

% initializing variables for cost and gradient to zero %
J = 0;
gradTheta1 = zeros(size(Theta1));
gradTheta2 = zeros(size(Theta2));
gradTheta3 = zeros(size(Theta3));

% feedforward implementation of neural network %
% Bias added to X %
a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = [ones(m,1) sigmoid(z3)];

z4 = a3 * Theta3';
a4 = sigmoid(z4);

finalPrediction = a4;

%-- Taking care of the NaN case, not sure if this is the right fix, but seems to work --%
finalPrediction(find(finalPrediction==1)) = 0.9999999999999999; % To avoid the NaN Case
finalPrediction(find(finalPrediction==0)) = 1.0000e-318; % To avoid the NaN Case

%-- For convenience, will remove later %
newy = y;

% Computing Cost without regularization %
J = (((newy(:))'*log(finalPrediction(:))) + ((1-newy(:))'*log(1-finalPrediction(:))))/(-m);

% Regularization part %
Theta1Reg = (Theta1(:,[2 : size(Theta1,2)])(:)'*Theta1(:,[2 : size(Theta1,2)])(:));
Theta2Reg = (Theta2(:,[2 : size(Theta2,2)])(:)'*Theta2(:,[2 : size(Theta2,2)])(:));
Theta3Reg = (Theta3(:,[2 : size(Theta3,2)])(:)'*Theta3(:,[2 : size(Theta3,2)])(:));

Regularization = (lambda/(2*m))*(Theta1Reg + Theta2Reg + Theta3Reg);


% Adding Regularization part to the cost %
J = J + Regularization;


% Implementation of BackPropagation Algorithm %

% Computation of Error in Layer 4 %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no of class labels (in this case - 1)%
% The dimensions have to be consistent with a4 %
Sdelta4 = a4 - newy;

% Computation of gradient for Theta 3 %
Cdelta3 = Sdelta4'*a3;

% Computation of Error in Layer 3 %
% The dimensions of this unit should be (m,n) where m = no. of examples, n = no. of units in layer3 + 1 - hlsize + 1 | the error for the bias unit is also computed, but isn't used further %
% The dimensions have to be consistent with a2 %
Sdelta3 = (Sdelta4*Theta3 .* [ones(m,1) sigmoidGradient(z3)]);

% Computation of gradient for Theta 2 %
Cdelta2 = Sdelta3(:,2:end)'*a2;

% Computation of Error in Layer 2 %
Sdelta2 = (Sdelta3(:,2:end)*Theta2 .* [ones(m,1) sigmoidGradient(z2)]);

% Computation of gradient for Theta 1 %
Cdelta1 = Sdelta2(:,[2 : end])'*a1;

% Averaged Gradient for Theta 1 %
gradTheta1 = Cdelta1/m;

% Averaged Gradient for Theta 2 %
gradTheta2 = Cdelta2/m;

% Averaged Gradient for Theta 3 %
gradTheta3 = Cdelta3/m;

% Computation of Regularization for Theta 1 %
regTheta1 = [zeros(size(Theta1,1),1) (lambda/m)*Theta1(:,[2:end])];

% Computation of Regularization for Theta 2 %
regTheta2 = [zeros(size(Theta2,1),1) (lambda/m)*Theta2(:,[2:end])];

% Computation of Regularization for Theta 3 %
regTheta3 = [zeros(size(Theta3,1),1) (lambda/m)*Theta3(:,[2:end])];

% Final regularized gradient of Theta1 %
gradTheta1 = gradTheta1 + regTheta1;

% Final regularized gradient of Theta2 %
gradTheta2 = gradTheta2 + regTheta2;

% Final regularized gradient of Theta3 %
gradTheta3 = gradTheta3 + regTheta3;

% Unrolling the computed gradients for use with advanced optimization techniques (fmincg) %
grad = [gradTheta1(:) ; gradTheta2(:) ; gradTheta3(:)];

end
