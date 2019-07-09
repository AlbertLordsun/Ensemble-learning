function w_c = CalculateClassifierWeight(label, feature, encode_classifier, CVO)

% encode_classifier (0-1 string): [SVM MLP GMM KNN LDA]
% encode_rule (0-1 string): [Product Sum Mean Max Meadian Minimum Weighted-Average]
% encode_classifer = '00010';
% encode_rule = '0001000';
% refer from : A novel classifer ensemble for recognition of multiple indoor air contaminants by an electronic nose
% w(i,j) is the accuracy of the ith classifier on the jth gas(错分类权重：对应分类器i对于将目标对象分类到j类上的权重)

d_c = length(encode_classifier);
d_y = 10;       %label class
w_c = zeros(d_c, d_y);
bestpre = zeros(size(label));

if (encode_classifier(1) == '1')
    % SVM
    this_classifier = 1;
    for i = 1 : CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
        ytest=label(teIdx,:);xtest=feature(teIdx,:);
        [~,bestc,bestg]=SVMcg(ytrain,xtrain,log2(10^5),1,log2(10^5),-9,2,0,5);
        option=['-s 0 -t 2',' -g ',num2str(bestg),' -c ',num2str(bestc),' -b 1 -q'];
        svmmodel=svmtrain(ytrain,xtrain,option);
        [bestpre(teIdx),~,~]=svmpredict(ytest,xtest,svmmodel, '-b 1'); %概率输出开关；%对应的三个输出是预测标签、准确度、标签概率
    end
    w_c(this_classifier,:) = norm_wc(bestpre, label, d_y);
end

if (encode_classifier(2) == '1')
    % MLP
    this_classifier = 2;
    for i = 1 : CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
        ytest=label(teIdx,:);xtest=feature(teIdx,:);
        py=bp(xtest, xtrain, ytrain);
        bestpre(teIdx) = py';
    end
    w_c(this_classifier,:) = norm_wc(bestpre, label, d_y);
end


if(encode_classifier(3)== '1')
    %PNN
    this_classifier = 3;
    for i = 1 : CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
        ytest=label(teIdx,:);xtest=feature(teIdx,:);
        net = super_newpnn(xtrain', ind2vec(ytrain'));
        temp_Y = sim(net,xtest');
        Y = vec2ind(temp_Y);
        bestpre(teIdx) = Y';
    end
    w_c(this_classifier,:) = norm_wc(bestpre,label,d_y);
end

if (encode_classifier(4) == '1')
    % KNN
    this_classifier = 4;
    for i = 1 : CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
        ytest=label(teIdx,:);xtest=feature(teIdx,:);
        bestcc = 0;
        for k = 1 : 20
            Factor = fitcknn(xtrain, ytrain, 'BreakTies', 'nearest', 'Distance', 'euclidean', 'NumNeighbors', k);
            temp= predict(Factor, xtest);
            acc = sum(temp==ytest)/length(ytest);
            if (bestcc < acc)
                bestcc = acc;
                bestpre(teIdx) = temp;
            end
        end
    end
    w_c(this_classifier,:) = norm_wc(bestpre, label, d_y);
end

if (encode_classifier(5) == '1')
    % LDA
    this_classifier = 5;
    for i = 1 : CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        ytrain=label(trIdx,:);xtrain=feature(trIdx,:);
        xtest=feature(teIdx,:);
        Mdl = fitcdiscr(xtrain, ytrain);
        bestpre(teIdx) = predict(Mdl, xtest);
    end
    w_c(this_classifier,:) = norm_wc(bestpre, label, d_y);
end

w_c = norm_class(w_c, d_y, d_c, encode_classifier);
end

function w = norm_class(w, d_y, d_c, encode_classifier)
idx = [];
for i = 1 : d_c
    if (encode_classifier(i) == '1')
        idx = [idx i]; %#ok<AGROW>
    end
end
for j = 1 : d_y
    mmin = min(w(idx,j));
    mmax = max(w(idx,j));
    if (mmin == mmax)
        continue;
    end
    w(idx,j) = (w(idx,j) - mmin) / (mmax - mmin);
end
end

function nrwc = norm_wc(bestpre, label, d_y)
% nrwc = zeros(1, d_y);
% sumwc = 0;
% for j = 1 : d_y
%     idx = label == j;
%     nrwc(j) = sum(bestpre(idx, :)==label(idx, :))/length(label(idx, :));
%     sumwc = sumwc + log(nrwc(j) / (1-nrwc(j)));
% end
% for j = 1 : d_y
%     nrwc(j) = log(nrwc(j) / (1-nrwc(j))) / sumwc;
% end

nrwc = zeros(1, d_y);
for j = 1 : d_y
    idx = (label == j);
    nrwc(j) = sum(bestpre(idx, :)==label(idx, :))/length(label(idx, :));
end
end