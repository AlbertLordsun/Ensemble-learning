% encode_classifier (0-1 string): [SVM MLP GP KNN LDA]
% encode_rule (0-1 string): [Product Sum Mean Max Meadian Minimum Weighted-Average]

clear; clc; close all;
load('M1std10dendrobeThenNorm.mat');
C30 = ['10000';'01000';'00100';'00010';'00001'];
T = 30;

for C = 1 : size(C30,1)
        feature = M1;
        nsample=size(feature,1);
        AccAdaBoost=zeros(10,1);  %������֤35�������ѵ���������ͨ��35�ν����Ѱ��optimal model��
        encode_classifer = C30(C,:);
        encode_rule = '0010000';
        Ans = zeros(10,1);        %35�ν�����֤�Ľ��
        py = zeros(size(label));
        
        % CVO = load('.\CVO\CVO1');
        CVO = cvpartition(label,'k',6);
        hwait = waitbar(0,'a');
        
        w_c = CalculateClassifierWeight(label, feature, encode_classifer, CVO);  %�õ��������Լ����л���������֮���Ȩ��
        w_c = w_c + 1e-5*ones(size(w_c));
        
        for rep = 1 : 2        %������֤��ѵ������
            for i = 1 : CVO.NumTestSets  %����������֤���̵�ѭ���������ǽ�������ʾ��Ȼ���ǻ�ȡѵ�������
%                 waitbar(i/CVO.NumTestSets,hwait,[encode_classifer ', ' num2str(T) ', ' num2str(rep) ' : ' num2str(i/CVO.NumTestSets*100)]);
                if (AccAdaBoost(i) == 1)
                    continue;
                end
                trIdx = CVO.training(i);  %������֤�е�ѵ���������Լ����䣻
                teIdx = CVO.test(i);
                ytrain=label(trIdx,:);xtrain=feature(trIdx,:);  %306*9��306*128
                ytest=label(teIdx,:);xtest=feature(teIdx,:);    
                [H, newT] = AdaBoostM2(ytrain, xtrain, encode_classifer, encode_rule, T, ytest, xtest,w_c);  %��õ���AdaboostM2�е����շ�����
                %     [H, w_c] = Bagging(ytrain, xtrain, encode_classifer, encode_rule, T, ytest, xtest);
                temp_py = AdaBoostClassify(xtest, H, encode_classifer, encode_rule, w_c, newT);
                Ans(i)=sum(temp_py==ytest)/length(ytest);
                if (Ans(i)>AccAdaBoost(i))
                    AccAdaBoost(i) = Ans(i);
                    py(teIdx) = temp_py;    %������֤�еķ�����
                end
            end
        end
        
        acc_cate = zeros(1,10);  %����������ڼ��ɷ���������ÿһ��ķ���׼ȷ��
        for i = 1 : 10
            acc_cate(i) = sum(py((i-1)*48+1:i*48) == i) / 48;  %�Ƚϵ��Ƕ�Ӧ���sample��������ԭ��label�Ƿ�ƥ��
        end
        
        save(['KIII-AdaBoost-' encode_classifer '-' encode_rule '-' num2str(T)], 'AccAdaBoost', 'py', 'acc_cate', 'encode_classifer', 'T');
        
%         close(hwait);
end
