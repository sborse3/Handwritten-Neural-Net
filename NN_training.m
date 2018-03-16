clc
clear all
close all

Test=load('test_data.mat');
Train=load('training_data.mat');
lambda=0.3;
T=@(x) -1*ones(1,length(x))+2./(1+exp(-lambda*x));
wqp=randn(49,4);
wpj=randn(49,49);
alpha=0.1;
str='training_';
ind=['M' 'E' '1' '7'];
rq=zeros(4,1);
Errs=[];
Err=10;
epochs=1;
tol=0.0001;
while epochs <=1000 && Err>=tol
    Err=0;
    for i=1:4
        rq(i)=1;
        for j=1:4
            str1=strcat(str,ind(i),num2str(j));
            I=getfield(Train,str1);
            I=I(:);
            %%%%%forward
            Op=T(I'*wpj)';
            Oq=T(Op'*wqp)';

            %%%%%backward
            dq=0.5*(rq-Oq).*(ones(4,1)+Oq).*(ones(4,1)-Oq);
            dwqp=alpha*Op*dq';
            wqpnew=wqp+dwqp;

            su=zeros(length(Op),1);
            for i1=1:4
                su=su+wqp(:,i1)*dq(i1);
            end
            dp=0.5*su.*(ones(49,1)+Op).*(ones(49,1)-Op);
            dwpj=alpha*I*dp';
            wpj=wpj+dwpj;
            wqp=wqpnew;
            Err=Err+0.5*sum((rq-Oq).^2);
        end
        %disp(rq)        
        rq=zeros(4,1);
    end
    disp('Epoch ')    
    disp(epochs)
    Errs=[Errs Err];
    epochs=epochs+1;
end

disp('Converges in epoch ')
disp(epochs-1)

figure;

plot(Errs)
title('Error vs Epochs')
xlabel('Epochs')
ylabel('Error')
save('NN_weights.mat','wpj','wqp');