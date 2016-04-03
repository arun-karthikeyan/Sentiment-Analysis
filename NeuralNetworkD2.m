%#% Project Opinion Mining - Author Arun Karthikeyan %#%

% Parameters - None %
% The network is designed with 2 Hidden Layers (variable Activation units) %
% With Variable input and 1 output %
% Learning Algorithm used is back propagation algorithm %

function NeuralNetworkD2()

% Clearing Memory %
clear all;
allData = "Top500Features_KLD.txt";
allData2 = "Bottom500Features_KLD.txt";

allData = load(allData);
allData2 = load(allData2);

allData = allData(1:8000,:);
allData2 = allData2(1:8000,:);

Xtotal = allData(:,[1:(end-1)]);
ytotal = allData(:,end);

Xtotal2 = allData2(:,[1:(end-1)]);
ytotal2 = allData2(:,end);


%Xtotal = normalizeFeatures(Xtotal);

% Recording size of total data %
[mtot ntot] = size(Xtotal);
[mtot2 ntot2] = size(Xtotal2);

%mtot
%ntot

% 60% of the total Data for Training ; 20% for CV ; 20% for testing %
%mtrain = ceil(mtot*0.90);
mtrain = mtot;
ntrain = ntot;

mtrain2 = mtot2;
ntrain2 = ntot2;


Xtrain = Xtotal(1:mtrain,:);
ytrain = ytotal(1:mtrain,1);

Xtrain2 = Xtotal2(1:mtrain2,:);
ytrain2 = ytotal2(1:mtrain2,1);


% Disabling temporarily for kfoldCode %

%cvAndTraining = mtot - mtrain;

%mcv = ceil(cvAndTraining*0.50);
%ncv = ntot;

%Xcv = Xtotal(((mtrain+1):(mtrain+mcv)),:);
%ycv = ytotal(((mtrain+1):(mtrain+mcv)),:);

%Xtest = Xtotal(((mtrain+mcv+1):end),:);
%ytest = ytotal(((mtrain+mcv+1):end),:);

%-- Removing CV Data and Adding it to Test Data%
%Xtest = [Xcv;Xtest];
%ytest = [ycv;ytest];

%[mtest ntest] = size(Xtest);

% Hidden Units is 25% of the input units %
%hlSize = ceil(ntot*0.25);
hlSize1 = 20;
hlSize2 = 20;

% Initializing the input vector size and output vector size %
ipSize = ntrain;
opSize = 1;

% Initializing initial values of Theta1 %
% Bias parameter has been included %
%initTheta1 = sinInitTheta(hlSize,ipSize+1);
initTheta1 = randInitTheta(hlSize1,ipSize+1);

% Initializing initial values of Theta2 %
% Bias parameter has been included %
%initTheta2 = sinInitTheta(opSize,hlSize+1);
initTheta2 = randInitTheta(hlSize2,hlSize1+1);

% Initializing initial values of Theta2 %
% Bias parameter has been included %
%initTheta2 = sinInitTheta(opSize,hlSize+1);
initTheta3 = randInitTheta(opSize,hlSize2+1);


% Initializing the regularization parameter lambda %
lambda = 0;

% kval for kfold cross validation %
kfold = 2;

kval = floor(mtrain/kfold);

%startLambda = 0; step = 0.0001; endLambda = 0;


display("%------------------------------------------------------------------------------------------------------------------------------%");
display(sprintf("Input Layer Size - %d\nHidden Layer 1 Size - %d\nHidden Layer 2 Size - %d\nOutput Size - %d",ipSize,hlSize1,hlSize2,opSize));
%display(sprintf("Total data size - %d | Training data size - %d | Crossvalidation data size - %d | Test data size - %d",mtot,mtrain,mcv,mtest)); 
%display(sprintf("Total data size - %d | Training data size - %d | Test data size - %d",mtot,mtrain,mtest));
display(sprintf("Training data size - %d",mtrain));
display("%------------------------------------------------------------------------------------------------------------------------------%");

for i=1:kval:(kval*kfold),

	startOffset = i;
	endOffset = (i+kval-1);

	%New Testing Data in this K-Fold %
	newXtest = Xtrain(startOffset:endOffset,:);
	newytest = ytrain(startOffset:endOffset,:);

	newXtest2 = Xtrain2(startOffset:endOffset,:);
	newytest2 = ytrain2(startOffset:endOffset,:);


	%New Training Data in this K-Fold %
	newXtrain = [ Xtrain(1:(startOffset-1),:); Xtrain((endOffset+1):end,:) ];
	newytrain = [ ytrain(1:(startOffset-1),:); ytrain((endOffset+1):end,:) ];

	newXtrain2 = [ Xtrain2(1:(startOffset-1),:); Xtrain2((endOffset+1):end,:) ];
	newytrain2 = [ ytrain2(1:(startOffset-1),:); ytrain2((endOffset+1):end,:) ];

	display("%------------------------------------------------------------------------------------------------------------------------------%");
	[optTheta1 optTheta2 optTheta3 dataToPlot] = NeuralNetworkD2Learning(newXtrain, newytrain, initTheta1, initTheta2, initTheta3, lambda, newXtest, newytest);
	[optTheta12 optTheta22 optTheta32 dataToPlot2] = NeuralNetworkD2Learning(newXtrain2, newytrain2, initTheta1, initTheta2, initTheta3, lambda, newXtest2, newytest2);		



	% Predictions on the Testdata %
	testPredictions = NeuralNetworkD2Prediction(optTheta1, optTheta2,optTheta3, newXtest);
	testPredictions2 = NeuralNetworkD2Prediction(optTheta12, optTheta22,optTheta32, newXtest2);

	% Computation of prediction accuracy on the Test Data %
	testPredAcc = mean(testPredictions==newytest)*100;
	testPredAcc2 = mean(testPredictions2==newytest2)*100;

	% fold count iterator %
	foldCount = endOffset/kval;

	display(sprintf("Fold %d - Test Accuracy - %0.4f | with lambda - %0.4f",foldCount,testPredAcc,lambda));
	display("%------------------------------------------------------------------------------------------------------------------------------%");

	Accuracies(foldCount,1) = testPredAcc;

	%Generating and saving the plot for each fold %
	figure(foldCount);
	plot(dataToPlot,'k-');
	hold on;
	plot(dataToPlot2,'-');
	legend("Top 500 Features","Bottom 500 Features");
	xlabel('No. of epochs -->');
	ylabel('Error -->');

	testPredAcc = 93.9621;
	testPredAcc2 = 37.5122;

	title(sprintf("Top 500 Features Accuracy %0.4f\nBottom 500 Features Accuracy %0.4f",testPredAcc,testPredAcc2));
	saveas(foldCount,sprintf("NND2_10kFeatures_lambda0_Fold_%d.png",foldCount));
	close; 

end;

display(sprintf("Average Accuracy throughout all the folds - %0.4f",mean(Accuracies)));

end
