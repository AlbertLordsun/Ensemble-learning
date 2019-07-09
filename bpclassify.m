function [netPerf,t,net]=bpclassify(newobj,inputs,targets,hiddenlayer)

temp_targets = zeros(10, size(targets,2));      %label class
for i = 1 : size(targets,2)
    temp_targets(targets(i), i) = 1;
end
net=patternnet(hiddenlayer,'trainscg');
net.trainParam.goal=1e-5;
net.trainParam.epochs=100;
net.trainParam.showWindow=false;
net.trainParam.showCommandline=false;
net.trainParam.show=10;
net.performFcn='mse';

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;


% Train the Network
[net,~] = train(net,inputs,temp_targets, 'useGPU', 'only', 'showResources', 'yes');

% Test the Network
outputs = net(inputs);
outputs=vec2ind(outputs);
netPerf=length(find(targets==outputs))/length(targets);
t=net(newobj);
t=vec2ind(t);
% performance = perform(net,targets,outputs);


