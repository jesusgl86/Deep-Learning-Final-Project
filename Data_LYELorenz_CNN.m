clear all
clc
%% Simulate pictures
dt=0.01; T=15; t=0:dt:T;
fs=100;
eRange = 10;
dim = 3;
ode_options = odeset('RelTol',10^(-10), 'AbsTol',10^(-11));
x0=[1,1,1];
sigma=0;
%% Making training data
N=1000;
Toutput=zeros(N,1);
XTrain=zeros(28,28,1,N);


for i=1:N
    i
    % Function
    b=8/3*rand; sig=10*rand; r=20*rand;
    Lorenz_sys = @(t,x)([ sig * (x(2) - x(1)) ; ...
    r * x(1)-x(1) * x(3) - x(2) ; ...
    x(1) * x(2) - b*x(3) ]);
    [t,y] = ode45(Lorenz_sys,t,x0);
    xdata = y(:,1);
%     plot(y(:,1))
%     figure
%     plot3(y(:,1),y(:,2),y(:,3))
    [~,lag] = phaseSpaceReconstruction(xdata,[],dim);
    ly=lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange);
    y=y+sigma*randn(size(t,1),3);
    xdatanoise=y(:,1);
    M=diffmat(xdatanoise);
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Toutput(i)=ly;
    XTrain(:,:,1,i)=I;
end
%%
imshow(I)
figure
plot3(y(:,1),y(:,2),y(:,3))
I=XTrain(:,:,1,1);
figure('visible','on');
imshow(I)

save('Ly_train_data.mat','Toutput','XTrain')
%% Making validation data
N=100;
Toutvalid=zeros(N,1);
Xvalid=zeros(28,28,1,N);

for i=1:N
    i
    %% Function
    b=8/3*rand; sig=10*rand; r=20*rand;
    Lorenz_sys = @(t,x)([ sig * (x(2) - x(1)) ; ...
    r * x(1)-x(1) * x(3) - x(2) ; ...
    x(1) * x(2) - b*x(3) ]);
    [t,y] = ode45(Lorenz_sys,t,x0);
    xdata = y(:,1);
    [~,lag] = phaseSpaceReconstruction(xdata,[],dim);
    ly=lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange);
    M=diffmat(xdata);
    %% plot
    y=y+sigma*randn(size(t,1),3);
    xdatanoise=y(:,1);
    M=diffmat(xdatanoise);
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Toutvalid(i)=ly;
    Xvalid(:,:,1,i)=I;
end
save('Ly_valid_data.mat','Toutvalid','Xvalid')
%% Making testing data
N=100;
Touttest=zeros(N,1);
Xtest=zeros(28,28,1,N);
time=0;
sigma=1;
for i=1:N
    i
    % Function
    b=8/3*rand; sig=10*rand; r=20*rand;
    Lorenz_sys = @(t,x)([ sig * (x(2) - x(1)) ; ...
    r * x(1)-x(1) * x(3) - x(2) ; ...
    x(1) * x(2) - b*x(3) ]);
    [t,y] = ode45(Lorenz_sys,t,x0);
    xdata = y(:,1);
    tic
    [~,lag] = phaseSpaceReconstruction(xdata,[],dim);
    ly=lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange);
    time=time+toc;
    y=y+sigma*randn(size(t,1),3);
    xdatanoise=y(:,1);
    M=diffmat(xdatanoise);
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Touttest(i)=ly;
    Xtest(:,:,1,i)=I;
end

avgRostime=time/(N)
save('Ly_test_data.mat','Touttest','Xtest')