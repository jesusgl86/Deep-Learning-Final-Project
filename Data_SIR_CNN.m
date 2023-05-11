clear all
clc
%% Simulate pictures
dt=.01; T=50;
t=0:dt:T;
ode_options = odeset('RelTol',10^(-10), 'AbsTol',10^(-11));
x0=[.99;.01;.1];
betamax=2; betamin=0;
%% Making training data
N1=1000;
sigma=0.1;
Toutput=zeros(N1,1);
XTrain=zeros(28,28,1,N1);
Nm=28;
for i=1:N1
    i
    %% Function
    beta=(betamax-betamin)*rand+betamin; mu=1/(365*50); gamma=1/28;
    SIR_sys=@(t,x) ([mu-beta*x(1)*x(2)-mu*x(1);...
    beta*x(1)*x(2)-mu*x(2)-gamma*x(2);...
    gamma*x(2)-mu*x(3)]);
    [t,y] = ode45(SIR_sys,t,x0);
    y(:,2)=y(:,2)+sigma*randn(size(y(:,2)));
    xdata = (y(:,2)-min(y(:,2)))/(max(y(:,2))-min(y(:,2)));
    Sigtrain0(i,:)=xdata;
    Toutput(i)=beta;
  
end

%% Making validation data
N2=100;
Toutvalid=zeros(N2,1);
Xvalid=zeros(28,28,1,N2);

for i=1:N2
    i
    %% Function
    beta=(betamax-betamin)*rand+betamin; mu=1/(365*50); gamma=1/28;
    SIR_sys=@(t,x) ([mu-beta*x(1)*x(2)-mu*x(1);...
    beta*x(1)*x(2)-mu*x(2)-gamma*x(2);...
    gamma*x(2)-mu*x(3)]);
    [t,y] = ode45(SIR_sys,t,x0);
    y(:,2)=y(:,2)+sigma*randn(size(y(:,2)));
    xdata = (y(:,2)-min(y(:,2)))/(max(y(:,2))-min(y(:,2)));
    Sigvalid0(i,:)=xdata;
    Toutvalid(i)=beta;

end

%% Making testing data
N3=100;
Touttest=zeros(N3,1);
Xtest=zeros(28,28,1,N3);

for i=1:N3
    i
    %% Function
    beta=(betamax-betamin)*rand+betamin; mu=1/(365*50); gamma=1/28;
    SIR_sys=@(t,x) ([mu-beta*x(1)*x(2)-mu*x(1);...
    beta*x(1)*x(2)-mu*x(2)-gamma*x(2);...
    gamma*x(2)-mu*x(3)]);
    [t,y] = ode45(SIR_sys,t,x0);
    y(:,2)=y(:,2)+sigma*randn(size(y(:,2)));
    xdata = (y(:,2)-min(y(:,2)))/(max(y(:,2))-min(y(:,2)));
    Sigtest0(i,:)=xdata;
    Touttest(i)=beta;
end

%%
maxval=max([max(Sigtest0) max(Sigvalid0) max(Sigtest0)]);
Sigtrain=Sigtrain0/maxval;
Sigvalid=Sigvalid0/maxval;
Sigtest=Sigtest0/maxval;
%%
for i=1:N1
    M=diffmat(Sigtrain(i,:));
    % plot
    I = mat2gray(M,[0 1]);
    I=imresize(I,[Nm Nm]);
    XTrain(:,:,1,i)=I;
    i
end
I=XTrain(:,:,1,1);
figure('visible','on');
imshow(I);
%%
for i=1:N2
    M=diffmat(Sigvalid(i,:));
    % plot
    I = mat2gray(M, [0 1]);
    I=imresize(I,[Nm Nm]);
    Xvalid(:,:,1,i)=I;  
    i
end


%%
for i=1:N3
    M=diffmat(Sigtest(i,:));
    % plot
    I = mat2gray(M,[0 1]);
    I=imresize(I,[Nm Nm]);
    Xtest(:,:,1,i)=I;
    i
end


save('SIR_train_data.mat','Toutput','XTrain')
save('SIR_valid_data.mat','Toutvalid','Xvalid')
save('SIR_test_data.mat','Touttest','Xtest')