%#% Project Opinion Mining - Author Arun Karthikeyan %#%
function LogisticRegression()

% Clearing Memory %
clear all;
allData = "Top10000Features_KLD.txt";

allData = load(allData);

%--%
%allData = allData(1:1010,:);

Xtotal = allData(:,[1:(end-1)]);
ytotal = allData(:,end);

%Xtotal = normalizeFeatures(Xtotal);

% Recording size of total data %
[mtot ntot] = size(Xtotal);

%Shuffling the data set %

idx = randperm(mtot);
Xtotal = Xtotal(idx,:);
ytotal = ytotal(idx,:);

% 60% of the total Data for Training ; 20% for CV ; 20% for testing %
mtrain = mtot;
%mtrain = ceil(mtot*0.80);
ntrain = ntot;

Xtrain = Xtotal(1:mtrain,:);
ytrain = ytotal(1:mtrain,1);

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

% randInitTheta = 1 signifies random Initialization of initial Values of Theta %
% randInitTheta = 0 signifies zeros Initialization of initial Values of Theta %
randInitTheta = 0;

%initializing lambda to be 0 %
lambda = 0;

% kval for kfold cross validation %
kfold = 10;

kval = floor(mtrain/kfold);

startLambda = 0; step = 0.0001; endLambda = 0;

for j=startLambda:step:endLambda,

for i=1:kval:(kval*kfold),
display("%------------------------------------------------------------------------------------------------------------------------------%");

startOffset = i;
endOffset = (i+kval-1);

%New Testing Data in this K-Fold %
newXtest = Xtrain(startOffset:endOffset,:);
newytest = ytrain(startOffset:endOffset,:);

%New Training Data in this K-Fold %
newXtrain = [ Xtrain(1:(startOffset-1),:); Xtrain((endOffset+1):end,:) ];
newytrain = [ ytrain(1:(startOffset-1),:); ytrain((endOffset+1):end,:) ];

%--%
%size(newXtest)
%size(newXtrain)

[Theta dataToPlot] = LogisticRegressionLearning(newXtrain,newytrain,j, newXtest, newytest,randInitTheta);
prediction = LogisticRegressionPrediction(newXtest,Theta);
predAccuracy = mean(prediction==newytest)*100;

% fold count iterator %
foldCount = endOffset/kval;

%Generating and saving the plot for each fold %
figure(foldCount);
plot(dataToPlot);
xlabel('No. of epochs -->');
ylabel('Error -->');
title(sprintf("Test Accuracy %0.4f",predAccuracy));
saveas(foldCount,sprintf("LR_10kFeatures_lambda0_Fold_%d.png",foldCount));
close; 

Accuracies(foldCount,1) = predAccuracy;

display(sprintf("Fold %d - Test Accuracy - %0.4f | with lambda - %0.4f",foldCount,predAccuracy,j));
display("%------------------------------------------------------------------------------------------------------------------------------%");
end;
%- This is assuming step size is 0.5-%
meanAcc((j*10000)+1,1) = mean(Accuracies);
display(sprintf("Average Accuracy throughout all the folds - %0.4f",meanAcc((j*10000)+1,1)));

end;

for k=startLambda:step:endLambda,
display(sprintf("Average Accuracy throughout all the folds - %0.4f with lambda - %0.4f",meanAcc((k*10000)+1,1),k));
end;

end
