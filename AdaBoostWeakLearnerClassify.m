function [py, FB] = AdaBoostWeakLearnerClassify(xtest, hypothesis, encode_classifier, encode_rule, w_c)

%������������ѵ���õ���������Լ������������DP

% encode_classifier (0-1 string): [SVM MLP GMM KNN LDA]
% encode_rule (0-1 string): [Product Sum Mean Max Meadian Minimum Weighted-Average]
% encode_classifer = '00010';
% encode_rule = '0001000';

d_c = length(encode_classifier);
d_y = 10;       %label class
d_s = size(xtest, 1);                                                      %�����������������������
DP = zeros(d_s, d_c, d_y);                                                 %�����������315*9���Ը÷�����ģ�Ͷ��ԣ�                                                 
FB = zeros(d_s, d_y);                                                      %����������*������ǩ

%�ȼ��������ʣ���˵��������������Ȩ�ص���
if (encode_classifier(1) == '1')
    % SVM
    svmmodel = hypothesis.SVM.svmmodel;
    [~,~,pos]=svmpredict(ones(d_s,1),xtest,svmmodel, '-b 1');              %pos������Ǳ�ǩ���������������-b��:�򿪸���������أ�               
    DP = RefreshDP(1,pos,DP,d_s,w_c,d_y);                                  %
end

if (encode_classifier(2) == '1')
    % MLP
    net = hypothesis.BP.net;
    pos = net(xtest');
    DP = RefreshDP(2,pos',DP,d_s,w_c,d_y);
end

if (encode_classifier(3) == '1')
    % PNN
    net = hypothesis.PNN.net;
    temp_Y = sim(net,xtest');
    for j = 1 : size(temp_Y,2)
        sumc = sum(temp_Y(:,j));
        temp_Y(:, j) = temp_Y(:, j) / sumc;
    end
    pos = temp_Y';
    DP = RefreshDP(3,pos,DP,d_s,w_c,d_y);
end

if (encode_classifier(4) == '1')
    % KNN
    [~, pos]= predict(hypothesis.KNN.Factor, xtest);                           
    DP = RefreshDP(4,pos,DP,d_s,w_c,d_y);
end

if (encode_classifier(5) == '1')
    % LDA
    [~,pos] = predict(hypothesis.LDA.Mdl, xtest);
    DP = RefreshDP(5,pos,DP,d_s,w_c,d_y);
end

py = zeros(d_s, 1);                                                        %py��xtest*1��
for i = 1 : d_s
    bestcc = 0;
    for j = 1 : d_y
        wc_pos = zeros(5,1);
        for k = 1 : 5
            wc_pos(k) = DP(i,k,j);
        end
        if (encode_rule(3) == '1')
            % Mean
            FB(i,j) = FB(i,j)  + mean(wc_pos);
        end
        
        if (encode_rule(4) == '1')
            % Max
            FB(i,j)  = FB(i,j)  + max(wc_pos);
        end
    end
end
FB = norm_pos(FB);
for i = 1 : d_s
    [~, py(i)] = max(FB(i,:));
end
end

function NewDP = RefreshDP(this,pos,DP,nSample,w_c,d_y)                    %DP�����������Ȩ���Ż�(nsample*9)
NewDP = DP;
pos = norm_pos(pos);
for i = 1 : nSample
    for j = 1 : d_y
        NewDP(i, this, j) = NewDP(i, this, j) + w_c(this,j) * pos(i,j);    
    end
end
end

function Newpos = norm_pos(pos)                                            %������ʵĹ�һ��(�й�һ������)
Newpos = zeros(size(pos));
for i = 1 : size(pos,1)
    Newpos(i, :) = pos(i,:) / sum(pos(i,:));
end
end