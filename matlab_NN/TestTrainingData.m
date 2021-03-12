inputs = 10*rand(3,50)-5
idealOutputs = (inputs(1,:)>0) & (inputs(2,:)>0) & (inputs(3,:)<0)

%%save('TestData.mat','inputs','idealOutputs');
writematrix(inputs)
writematrix(idealOutputs)