function t=bp(newobj,feature,label)

m=size(feature,2);
num=[sqrt(m+1)+1 sqrt(m+1)+10 log2(m)];
minNum=ceil(min(num));
maxNum=floor(max(num));
bestacc=0;
for hiddenLayer=minNum:maxNum
    [netPerf,tmp,~]=bpclassify(newobj',feature',label',hiddenLayer);  
    disp(num2str(hiddenLayer));
    if netPerf>bestacc   
        bestacc=netPerf;
        t=tmp;
    end
end