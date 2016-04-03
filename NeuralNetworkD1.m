%#% Project Opinion Mining - Author Arun Karthikeyan %#%

% Parameters - None %
% The network is designed with 1 Hidden Layer (variable Activation units) %
% With Variable input and 1 output %
% Learning Algorithm used is back propagation algorithm %

function NeuralNetworkD1()

% Clearing Memory %
clear all;

allData = "Top100Features_KLD.txt";

allData = load(allData);

%--%
allData = allData(1:8000,:);

Xtotal = allData(:,[1:(end-1)]);
ytotal = allData(:,end);

Xtotal = normalizeFeatures(Xtotal);

% Recording size of total data %
[mtot ntot] = size(Xtotal);

mtot
ntot

% 60% of the total Data for Training ; 20% for CV ; 20% for testing %
mtrain = ceil(mtot*0.90);
ntrain = ntot;

Xtrain = Xtotal(1:mtrain,:);
ytrain = ytotal(1:mtrain,1);

cvAndTraining = mtot - mtrain;

mcv = ceil(cvAndTraining*0.50);
ncv = ntot;

Xcv = Xtotal(((mtrain+1):(mtrain+mcv)),:);
ycv = ytotal(((mtrain+1):(mtrain+mcv)),:);

Xtest = Xtotal(((mtrain+mcv+1):end),:);
ytest = ytotal(((mtrain+mcv+1):end),:);

%-- Removing CV Data and Adding it to Test Data%
Xtest = [Xcv;Xtest];
ytest = [ycv;ytest];

[mtest ntest] = size(Xtest);

% Hidden Units is 25% of the input units %
%hlSize = ceil(ntot*0.25);
hlSize = 30	;

% Initializing the input vector size and output vector size %
ipSize = ntrain;
opSize = 1;

% Initializing initial values of Theta1 %
% Bias parameter has been included %
%initTheta1 = sinInitTheta(hlSize,ipSize+1);
initTheta1 = randInitTheta(hlSize,ipSize+1);

% Initializing initial values of Theta2 %
% Bias parameter has been included %
%initTheta2 = sinInitTheta(opSize,hlSize+1);
initTheta2 = randInitTheta(opSize,hlSize+1);


% Initializing the regularization parameter lambda %
lambda = 0;

% Initial prediction with initial set of parameters on the entire input set %
initPredictions = NeuralNetworkD1Prediction(initTheta1, initTheta2, Xtotal);

% Initial prediction with initial set of parameters on the test set %
initTestPreds = NeuralNetworkD1Prediction(initTheta1,initTheta2,Xtest);

% Initial prediction with initial set of parameters on the cross validation set %
initCVPreds = NeuralNetworkD1Prediction(initTheta1,initTheta2,Xcv);

% Initial prediction accuracy on the entire input set %
initPredAcc = mean(initPredictions==ytotal)*100;

% Initial prediction accuracy on the test set %
initTestPredAcc = mean(initTestPreds==ytest)*100;

% Initial predictino accuracy on the cross validation set %
initCVPredAcc = mean(initCVPreds==ycv)*100;

display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("Input Layer Size - %d\nHidden Layer Size - %d\nOutput Size - %d",ipSize,hlSize,opSize));
%display(sprintf("Total data size - %d | Training data size - %d | Crossvalidation data size - %d | Test data size - %d",mtot,mtrain,mcv,mtest)); 
display(sprintf("Total data size - %d | Training data size - %d | Test data size - %d",mtot,mtrain,mtest));
display("%------------------------------------------------------------------------------------------------------------------------------%");

for i=0:0.25:2,
	[optTheta1 optTheta2 dataToPlot] = NeuralNetworkD1Learning(Xtrain, ytrain, initTheta1, initTheta2, i, Xtest, ytest);		

	% Predictions on the Testdata %
	testPredictions = NeuralNetworkD1Prediction(optTheta1, optTheta2, Xtest);

	% Computation of prediction accuracy on the Test Data %
	testPredAcc = mean(testPredictions==ytest)*100;

	display(sprintf("Test Accuracy - %0.4f | with lambda - %0.4f",testPredAcc,i));
	display("%------------------------------------------------------------------------------------------------------------------------------%");
end;

end
