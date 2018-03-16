clc
clear all
close all

load('NN_weights.mat')

Test=load('test_data.mat');
lambda=0.3;
T=@(x) -1*ones(1,length(x))+2./(1+exp(-lambda*x));
ind=['M' 'E' '1' '7'];

str2='test_';
for i=1:4
    str3=strcat(str2,ind(i));
    I=getfield(Test,str3);
    I=I(:);
    
    %%%%%forward    
    Op=T(I'*wpj)';
    Oq=T(Op'*wqp)';
    disp(str3)
    disp(Oq);
end
