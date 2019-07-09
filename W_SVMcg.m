function [bestacc,bestc,bestg] = W_SVMcg(train_label,train,cmin,cstep,cmax,gmin,gstep,gmax,w)
%SVMcg cross validation by llf

CVO = cvpartition(train_label, 'k', 5);
py = zeros(size(train_label));

%% about the parameters of SVMcg
if nargin < 9
    v=3;
end
if nargin < 6
    v = 3;
    gmax = 5;
    gmin = -5;
    gstep = 1;
end
if nargin < 3
    v = 3;
    gmax = 5;
    gmin = -5;
    gstep = 1;
    cmax = 5;
    cmin = -5;
    cstep=1;
end
%% X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 0;
bestg = 0;
bestacc = 0;
basenumg = 2;
basenumc=2;
for i = 1:m
    for j = 1:n
        cmd = ['-s 0 -t 2 -g ',num2str( basenumg^Y(i,j) ),' -c ',num2str( basenumc^X(i,j) ),' -q'];
        for r = 1 : CVO.NumTestSets
            trIdx = CVO.training(r);
            teIdx = CVO.test(r);
            svmmodel = svmtrain(train_label(trIdx,:), train(trIdx,:), cmd);
            [py(teIdx,:),~,~]=svmpredict(train_label(teIdx,:),train(teIdx,:),svmmodel);
        end
        cg(i,j)=sum(w(py==train_label));
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenumc^X(i,j);
            bestg = basenumg^Y(i,j);
        end
        if ( cg(i,j) == bestacc && bestc > basenumc^X(i,j) )
            bestacc = cg(i,j);
            bestc = basenumc^X(i,j);
            bestg = basenumg^Y(i,j);
        end
        
    end
end
%% to draw the acc with different c & g
% [C,h] = contour(X,Y,cg,60:accstep:100);
% clabel(C,h,'FontSize',10,'Color','r');
% xlabel('log2c','FontSize',10);
% ylabel('log2g','FontSize',10);
% grid on;