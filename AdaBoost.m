function [H, w_c, t] = AdaBoost(label, feature, encode_classifer, encode_rule, T, ytest, xtest)

% hypothesis.SVM.svmmodel = svmmodel;
% hypothesis.BP.net=net;
% hypothesis.GMM.gmd = gmd;
% hypothesis.KNN.Factor=Factor;
% hypothesis.LDA.Mdl = Mdl;

nSample = size(feature, 1);
w = ones(nSample,1)/nSample;
beta = zeros(1, T);
acc_train = zeros(T,1);
acc_test = zeros(T,1);

% w_c: the weights of classifier * class
w_c = CalculateClassifierWeight(label, feature, encode_classifer, ytest, xtest);

t = 1;
while (t<=T)
    disp(t);
    % 1. Select a training data subset S(t), drawn from the distribution w(t)
    [xtrain, ytrain, wtrain] = SubsetSearch(label, feature, w);
    % 2. Train WeakLearner with S(t), receive hypothesis h(t)
    hypothesis(t) = TrainWeakLearner(xtrain, ytrain, encode_classifer, wtrain); %#ok<*AGROW>
    % 3. Calculate the error of h(t)
    h = AdaBoostWeakLearnerClassify(feature, hypothesis(t), encode_classifer, encode_rule, w_c);
    err = sum(w(h~=label));
    if (err>0.5)
        w = ones(nSample,1)/nSample;
        continue;
    end
    % 4. Set beta=e/(1-e)
    beta(t) = (err+1e-5)/(1-err);
    % 5. Update distribution
    for i = 1 : numel(label)
        if h(i) == label(i)
            DR = beta(t);
        else
            DR = 1;
        end
        w(i) = w(i) * DR;
    end
    w = w / sum(w);
    H(t).beta = beta(t);
    H(t).hypothesis = hypothesis(t);
    
%     pytesttrain = AdaBoostClassify(feature, H, encode_classifer, encode_rule, w_c, t);
%     pytest = AdaBoostClassify(xtest, H, encode_classifer, encode_rule, w_c, t);
%     acc_train(t) = sum(pytesttrain == label) / length(label) * 100;
%     acc_test(t) = sum(pytest == ytest) / length(ytest) * 100;
%     disp([num2str(acc_train(t)) '  ' num2str(acc_test(t))]);
%     figure(1);
%     drawnow;
%     plot(acc_train(1:t),'DisplayName','acc_train', 'color', 'r');hold all;plot(acc_test(1:t),'DisplayName','acc_test', 'color', 'k');hold off;
%     xlabel(['t = ' num2str(t)]);
%     
    if (err==0)
        w = ones(nSample,1)/nSample;
    end
    
    t = t + 1;
end
end