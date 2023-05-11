clear all
clc
%% Simulate pictures
n=1000;
a=0;
b=10;
vx=linspace(a,b,n+1); 
sigma=0;
g=@(ra) (exp(ra*vx))+sigma*randn(size(vx));
r0=-1; r1=1;
%% Making training data
N=1000;
Toutput=zeros(N,1);
XTrain=zeros(28,28,1,N);
%%
for i=1:N
    i
    % Function
    %r=2*(rand-.5);
    r=(r1-r0)*rand+r0;
    M=diffmat(g(r));
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    %I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
    Toutput(i)=r;
    XTrain(:,:,1,i)=I;
end
figure('visible','on'),imshow(XTrain(:,:,1,1))
Toutput(1)
save('Exp_train_data.mat','Toutput','XTrain','r0','r1')
%% Making validation data
N=100;
Toutvalid=zeros(N,1);
Xvalid=zeros(28,28,1,N);
%%
for i=1:N
    i
    % Function
    r=(r1-r0)*rand+r0;
    M=diffmat(g(r));
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Toutvalid(i)=r;
    Xvalid(:,:,1,i)=I;
end
save('Exp_valid_data.mat','Toutvalid','Xvalid')
%% Making testing data
N=100;
Touttest=zeros(N,1);
Xtest=zeros(28,28,1,N);
%%
for i=1:N
    i
    % Function
    r=(r1-r0)*rand+r0;
    M=diffmat(g(r));
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Touttest(i)=r;
    Xtest(:,:,1,i)=I;
end
save('Exp_test_data.mat','Touttest','Xtest')