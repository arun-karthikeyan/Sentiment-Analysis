%#% Project Opinion Mining - Author Arun Karthikeyan %#%

function [optTheta1 optTheta2 optTheta3 dataToPlot] = NeuralNetworkD2Learning(X, y, initTheta1, initTheta2, initTheta3, lambda, Xtest, ytest)

% Adding the Maximum Iterations for Learning using fmincg in the options parameter %
options  = optimset('MaxIter', 100);
display("Training for 100 epochs");

hlSize1 = size(initTheta1,1);
hlSize2 = size(initTheta2,1);
ipSize = size(initTheta1,2)-1; 
hlSize = [hlSize1 ; hlSize2];

initVectorTheta = [ initTheta1(:) ; initTheta2(:) ; initTheta3(:) ];

costFunction = @(t) NeuralNetworkD2Cost(t, ipSize, hlSize1, hlSize2, X, y, lambda);

[optVectorTheta cost iterations dataToPlot] = fmincgD2(costFunction, initVectorTheta, options, Xtest, ytest, hlSize, ipSize);

optTheta1 = reshape(optVectorTheta(1:hlSize1 * (ipSize + 1)), hlSize1, (ipSize + 1));
optTheta2 = reshape(optVectorTheta((1 + (hlSize1 * (ipSize + 1))):((hlSize1 * (ipSize + 1)) + (hlSize2 * (hlSize1 + 1)))), hlSize2, (hlSize1 + 1));
optTheta3 = reshape(optVectorTheta((((hlSize1 * (ipSize + 1)) + (hlSize2 * (hlSize1 + 1)))+1) : end),1,hlSize2+1);

end

