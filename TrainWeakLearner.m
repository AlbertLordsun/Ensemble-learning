function hypothesis = TrainWeakLearner(xtrain, ytrain, encode_classifer, w)

%��ͨ��10�۽�����֤ѵ����������������������ѵ���õ���Ӧ�Ļ�������ģ�ͽ��
CVO = cvpartition(ytrain, 'k', 10);

if (encode_classifer(1) == '1')
    % SVM����-s����C-SVC���ͣ���-g����RBF�˺������ͣ���-g�����������ݵ�����ֵ����-c�����˺����еĳͷ���������-b��������������أ���
    [~,bestc,bestg]=W_SVMcg(ytrain,xtrain,log2(10^5),1,log2(10^5),-9,2,0,w);
    option=['-s 0 -t 2',' -g ',num2str(bestg),' -c ',num2str(bestc),' -b 1 -q'];
    svmmodel=svmtrain(ytrain,xtrain,option);
    hypothesis.SVM.svmmodel = svmmodel;
end

if (encode_classifer(2) == '1')
    % MLP
    m=size(xtrain,2);
    num=[sqrt(m+1)+1 sqrt(m+1)+10 log2(m)];
    minNum=ceil(min(num));
    maxNum=floor(max(num));
    bestacc=0;
    for hiddenLayer=minNum:maxNum
        [~,py,net]=bpclassify(xtrain',xtrain',ytrain',hiddenLayer);
        py = py';
        netPerf = sum(w(py==ytrain));
        disp(num2str(hiddenLayer));
        if (netPerf > bestacc)
            bestacc=netPerf;
            hypothesis.BP.net=net;
        end
    end
end

if (encode_classifer(3) == '1')
    % PNN
    net = super_newpnn(xtrain', ind2vec(ytrain'));
    hypothesis.PNN.net = net;
end

if (encode_classifer(4) == '1')
    % KNN
    bestcc = 0;
    temp = zeros(size(ytrain));
    for k = 1 : 20
        for i = 1 : CVO.NumTestSets
            trIdx = CVO.training(i);
            teIdx = CVO.test(i);
            temp_ytrain=ytrain(trIdx,:);temp_xtrain=xtrain(trIdx,:);
            temp_xtest=xtrain(teIdx,:);
            Factor = fitcknn(temp_xtrain, temp_ytrain, 'BreakTies', 'nearest', 'Distance', 'euclidean', 'NumNeighbors', k);
            temp(teIdx,:)= predict(Factor, temp_xtest);
        end
        acc = sum(w(temp==ytrain));
        if (acc > bestcc)
            bestcc = acc;
            hypothesis.KNN.Factor=Factor;
        end
    end
end

if (encode_classifer(5) == '1')
    % LDA
    Mdl = fitcdiscr(xtrain, ytrain, 'Weights', w);
    hypothesis.LDA.Mdl = Mdl;
end