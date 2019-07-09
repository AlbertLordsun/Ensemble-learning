function [H, newT] = AdaBoostM2(label, feature, encode_classifer, encode_rule, T, ytest, xtest, w_c)

% 训练AdaboostM2模型结果（相应的算法框架结构）

% hypothesis.SVM.svmmodel = svmmodel;
% hypothesis.BP.net=net;
% hypothesis.GMM.gmd = gmd;
% hypothesis.KNN.Factor=Factor;
% hypothesis.LDA.Mdl = Mdl;

[nSample, ~] = size(feature);  %315*128
nType = numel(unique(label));  %9
[q,D,w] = Initialization(nSample, nType,label);  %315；9；315*1；
acc_train = zeros(T,1);                          %迭代过程中训练的准确度
acc_test = zeros(T,1);                           %迭代过程中测试的准确度
newT = T;
vote_vl1 = zeros(size(feature,1),10);       %label class
vote_vl2 = zeros(size(xtest,1),10);         %label class
%testresult = zeros(T,10);                        %迭代过程中测试结果（为何是乘以10的？）

% w_c: the weights of classifier * class

t = 1;
while (t<=T)
    disp(t);
    %参数设置：系统中的样本分布设置、分类情况的标签加权函数
    for i =1 : nSample
        W = sum(w(i,:));                                                   %每行分类标签的权重组合
        for j = 1 : nType                                                  %标签加权函数设置
            if (j == label(i))
                q(i,j) = 0;
                continue;
            end
            q(i,j) = w(i,j)/W;
        end
    end
    cntW = sum(sum(w));
    for i = 1 : nSample                                                    %样本分布的调整（有利于重点分析难分类或是错分类的样本数据）
        D(i) = sum(w(i,:))/cntW;
    end
    %根据样本权重D选择训练子集；(更新了训练样本数据集的权重)
    [xtrain, ytrain, wtrain] = SubsetSearch(label, feature, D);
    %训练弱分类器，并获得分类假设结果
    hypothesis(t) = TrainWeakLearner(xtrain, ytrain, encode_classifer, wtrain); %#ok<*AGROW>
    %训练样本特征集合所得到的矩阵B（样本数量*标签）
    [~, B] = AdaBoostWeakLearnerClassify(feature, hypothesis(t), encode_classifer, encode_rule, w_c);
    
    err = 0;
    for i =1 : nSample                                                     %误差修正
        err = err + 0.5*D(i)*( 1-B(i,label(i))+sum(q(i,:).*B(i,:)) );
    end
    if (err>0.5)
                [q,D,w] = Initialization(nSample, nType,label);
                continue;
%         newT = t-1;
%         break;
    end
    
    beta = (err+1e-5)/(1-err);                                             %误差归一beta修正
    
    for i = 1 : nSample                                                    %具体分类情况的权重修正
        for j =1 : nType
            w(i,j) = w(i,j) * beta^(0.5*( 1+ B(i,label(i)) - (sum(B(i,:)) - B(i,label(i))) ));
        end
    end
    H(t).beta = beta;                                                      %这两行结果有什么作用？返回参数？
    H(t).hypothesis = hypothesis(t);
    
    %迭代次数优化（合适的迭代次数可尽量减少系统的过拟合）——通过分析变化曲线可以分析获得较优的分析结果
    [pytesttrain, vote_vl1] = AdaBoostClassify_Debug(feature, H, encode_classifer, encode_rule, w_c, t, vote_vl1);
    [pytest, vote_vl2] = AdaBoostClassify_Debug(xtest, H, encode_classifer, encode_rule, w_c, t, vote_vl2);
    acc_train(t) = sum(pytesttrain == label) / length(label) * 100;        %训练用的是315
    acc_test(t) = sum(pytest == ytest) / length(ytest) * 100;              %测试用的是9个样本（可用来分析过拟合）
    disp([num2str(acc_train(t)) '  ' num2str(acc_test(t))]);
    figure(1);
    drawnow;
    plot(acc_train(1:t),'DisplayName','acc_train', 'color', 'r');hold all;plot(acc_test(1:t),'DisplayName','acc_test', 'color', 'k');hold off;
    xlabel(['t = ' num2str(t)]);
    
    if (err<1e-5)
        [q,D,w] = Initialization(nSample, nType,label);                    %误差过小初始化
%         newT = t;
%         break;
    end
    
    t = t + 1;
end
end

function [q,D,w] = Initialization(nSample, nType,label)                    %（315,9,315）
q = ones(nSample, nType)/(nType-1);                                        %标签加权函数q
D = ones(nSample, 1)/nSample;                                              %样本分布D
w = ones(nSample, nType)/nSample/(nType-1);                                %样本误分类权重w
for i = 1 : nSample
    q(i,label(i)) = 0;
    w(i,label(i)) = 0;
end
end
