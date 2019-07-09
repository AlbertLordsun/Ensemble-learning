function [H, newT] = AdaBoostM2(label, feature, encode_classifer, encode_rule, T, ytest, xtest, w_c)

% ѵ��AdaboostM2ģ�ͽ������Ӧ���㷨��ܽṹ��

% hypothesis.SVM.svmmodel = svmmodel;
% hypothesis.BP.net=net;
% hypothesis.GMM.gmd = gmd;
% hypothesis.KNN.Factor=Factor;
% hypothesis.LDA.Mdl = Mdl;

[nSample, ~] = size(feature);  %315*128
nType = numel(unique(label));  %9
[q,D,w] = Initialization(nSample, nType,label);  %315��9��315*1��
acc_train = zeros(T,1);                          %����������ѵ����׼ȷ��
acc_test = zeros(T,1);                           %���������в��Ե�׼ȷ��
newT = T;
vote_vl1 = zeros(size(feature,1),10);       %label class
vote_vl2 = zeros(size(xtest,1),10);         %label class
%testresult = zeros(T,10);                        %���������в��Խ����Ϊ���ǳ���10�ģ���

% w_c: the weights of classifier * class

t = 1;
while (t<=T)
    disp(t);
    %�������ã�ϵͳ�е������ֲ����á���������ı�ǩ��Ȩ����
    for i =1 : nSample
        W = sum(w(i,:));                                                   %ÿ�з����ǩ��Ȩ�����
        for j = 1 : nType                                                  %��ǩ��Ȩ��������
            if (j == label(i))
                q(i,j) = 0;
                continue;
            end
            q(i,j) = w(i,j)/W;
        end
    end
    cntW = sum(sum(w));
    for i = 1 : nSample                                                    %�����ֲ��ĵ������������ص�����ѷ�����Ǵ������������ݣ�
        D(i) = sum(w(i,:))/cntW;
    end
    %��������Ȩ��Dѡ��ѵ���Ӽ���(������ѵ���������ݼ���Ȩ��)
    [xtrain, ytrain, wtrain] = SubsetSearch(label, feature, D);
    %ѵ����������������÷��������
    hypothesis(t) = TrainWeakLearner(xtrain, ytrain, encode_classifer, wtrain); %#ok<*AGROW>
    %ѵ�����������������õ��ľ���B����������*��ǩ��
    [~, B] = AdaBoostWeakLearnerClassify(feature, hypothesis(t), encode_classifer, encode_rule, w_c);
    
    err = 0;
    for i =1 : nSample                                                     %�������
        err = err + 0.5*D(i)*( 1-B(i,label(i))+sum(q(i,:).*B(i,:)) );
    end
    if (err>0.5)
                [q,D,w] = Initialization(nSample, nType,label);
                continue;
%         newT = t-1;
%         break;
    end
    
    beta = (err+1e-5)/(1-err);                                             %����һbeta����
    
    for i = 1 : nSample                                                    %������������Ȩ������
        for j =1 : nType
            w(i,j) = w(i,j) * beta^(0.5*( 1+ B(i,label(i)) - (sum(B(i,:)) - B(i,label(i))) ));
        end
    end
    H(t).beta = beta;                                                      %�����н����ʲô���ã����ز�����
    H(t).hypothesis = hypothesis(t);
    
    %���������Ż������ʵĵ��������ɾ�������ϵͳ�Ĺ���ϣ�����ͨ�������仯���߿��Է�����ý��ŵķ������
    [pytesttrain, vote_vl1] = AdaBoostClassify_Debug(feature, H, encode_classifer, encode_rule, w_c, t, vote_vl1);
    [pytest, vote_vl2] = AdaBoostClassify_Debug(xtest, H, encode_classifer, encode_rule, w_c, t, vote_vl2);
    acc_train(t) = sum(pytesttrain == label) / length(label) * 100;        %ѵ���õ���315
    acc_test(t) = sum(pytest == ytest) / length(ytest) * 100;              %�����õ���9����������������������ϣ�
    disp([num2str(acc_train(t)) '  ' num2str(acc_test(t))]);
    figure(1);
    drawnow;
    plot(acc_train(1:t),'DisplayName','acc_train', 'color', 'r');hold all;plot(acc_test(1:t),'DisplayName','acc_test', 'color', 'k');hold off;
    xlabel(['t = ' num2str(t)]);
    
    if (err<1e-5)
        [q,D,w] = Initialization(nSample, nType,label);                    %����С��ʼ��
%         newT = t;
%         break;
    end
    
    t = t + 1;
end
end

function [q,D,w] = Initialization(nSample, nType,label)                    %��315,9,315��
q = ones(nSample, nType)/(nType-1);                                        %��ǩ��Ȩ����q
D = ones(nSample, 1)/nSample;                                              %�����ֲ�D
w = ones(nSample, nType)/nSample/(nType-1);                                %���������Ȩ��w
for i = 1 : nSample
    q(i,label(i)) = 0;
    w(i,label(i)) = 0;
end
end
