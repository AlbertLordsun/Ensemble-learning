% clear;
% CVO = load('CVO35');
% CVO = CVO.CVO;
% load('Enose.mat');
% feature = Enose;

load('M2stdSNE10.mat');
feature=mappedX;
CVO=cvpartition(label,'k',10);

% to aviod error Ill-conditioned covariance created at iteration 2.
% 1.Pre-process your data to remove correlated features.
% 2.Set 'SharedCov' to true to use an equal covariance matrix for every component.
% 3.Set 'CovType' to 'diagonal'.
% 4.Use 'Regularize' to add a very small positive number to the diagonal of every covariance matrix.
% 5.Try another set of initial values.

py = zeros(size(label));
pos = zeros(size(label,1), 10);
mu = zeros(10, 128);
sigma = eye(128, 128);
pw = ones(1,10)/10;

for i = 1: 10
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
    ytest=label(teIdx,:);xtest=feature(teIdx,:);
    
% 	[~,bestc,bestg]=SVMcg(ytrain,xtrain,log2(10^5),1,log2(10^5),-9,2,0,5);
% 	option=['-s 0 -t 2',' -g ',num2str(bestg),' -c ',num2str(bestc),' -b 1 -q'];
% 	svmmodel=svmtrain(ytrain,xtrain,option);
% 	py(teIdx)=svmpredict(ytest,xtest,svmmodel, '-b 1');
    
% 	py(teIdx)=bp(xtest, xtrain, ytrain);
    
% 	py(teIdx) = classify(xtest, xtrain, ytrain, 'linear');

% 	Factor = fitcknn(xtrain, ytrain, 'BreakTies', 'nearest', 'Distance', 'euclidean', 'NumNeighbors', 2);
% 	py(teIdx)= predict(Factor, xtest);

    net = super_newpnn(xtrain', ind2vec(ytrain'));
    temp_Y = sim(net,xtest');
    Y = vec2ind(temp_Y);
    py(teIdx) = Y';

    Acc(i) = sum(py(teIdx) == ytest) / length(ytest); %#ok<*SAGROW>
end

acc_cate = zeros(10,1);
for i = 1 : 10
    acc_cate(i,1) = sum(py((i-1)*48+1:i*48) == i) / 48;
end