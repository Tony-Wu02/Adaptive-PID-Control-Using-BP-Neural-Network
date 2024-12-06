function [sys,x0,str,ts]=my_exppidf(t,x,u,flag)
switch flag,
    case 0,
        [sys,x0,str,ts]=mdlInitializeSizes;
    case 2,
        sys=mdlUpdates(x,u);
    case 3,
        sys=mdlOutputs(t,x,u);
    case {1,4,9},
        sys=[];
    otherwise
        error(['unhandled flag=',num2str(flag)]);%Exception handling
end
function[sys,x0,str,ts]=mdlInitializeSizes
    sizes=simsizes;%The structure used to set module parameters is generated using simsizes
    sizes.NumContStates=0;%The number of continuous state variables of the module
    sizes.NumDiscStates=3;%The number of discrete state variables of the module
    sizes.NumOutputs=4;%The number of module output variables
    sizes.NumInputs=7;%Number of module input variables
    sizes.DirFeedthrough=1;%Whether the module has direct connection, 1 means there is direct connection, if it is 0, there cannot be u in the mdlOutputs function
    sizes.NumSampleTimes=1;%The number of sampling times of the module is at least one
    sys=simsizes(sizes);%After setting, assign it to sys output
    x0=zeros(3,1);%System status variable settings
    str=[];
    ts=[0 0];%The sampling period is set to 0 to indicate a continuous system.
             %ts=[0.001 0];% The sampling period is set to 0 to indicate a continuous system.
function sys=mdlUpdates(x,u)
        T=0.001;
        x=[u(5); x(2)+u(5)*T; (u(5)-u(4))/T];%3 state quantities (deviation, deviation sum and deviation change), u(5) is the deviation, u(4) is the previous deviation, and x(2) is the previous deviation sum
        sys=[x(1);x(2);x(3)];
function sys=mdlOutputs(t,x,u)
            xite=0.2;
            alfa=0.05;
            IN=3; H=5; OUT=3;
            wi=rand(5,3);%Generate a 5*3 random number matrix, the random numbers are in the range (0, 1)
            wi_1=wi;wi_2=wi;wi_3=wi;
            wo=rand(3,5);
            wo_1=wo;wo_2=wo;wo_3=wo;
            Oh=zeros(5,1);%Generate a 1*5 zero matrix (row matrix)
            I=Oh;
            xi=[u(1),u(3),u(5)];%Three inputs for neural network training: expected value, error, and actual value
            epid=[x(1);x(2);x(3)];%3 state variables (bias, sum of deviations, deviation change) (3*1 matrix, column vector)
            I=xi*wi';%Hidden layer input
            for j=1:1:5
                Oh(j)=(exp(I(j))-exp(-I(j)))/(exp(I(j))+exp(-I(j)));%Output value of hidden layer (1*5 matrix) row matrix
            end
            K1=wo*Oh;%Input of the output layer (3*1 matrix)
            for i=1:1:3
                K(i)=exp(K1(i))/(exp(K1(i))+exp(-K1(i)));%Get the output of the output layer (KP, KI, KD) (1*3 matrix, row vector)
            end
            u_k=K*epid;%Calculate the control law u, 1 value
            %The following are the weight adjustments
            %Weight adjustment from hidden layer to output layer
            dyu=sign((u(3)-u(2))/(u(7)-u(6)+0.0001));
            for j=1:1:3
                dK(j)=2/(exp(K1(j))+exp(-K1(j)))^2; %The first-order derivative of the output layer
            end
            for i=1:1:3
                delta3(i)=u(5)*dyu*epid(i)*dK(i);  %delta of the output layer
            end
            for j=1:1:3
                for i=1:1:5
                    d_wo=xite*delta3(j)*Oh(i)+alfa*(wo_1-wo_2);
                end
            end
            wo=wo_1+d_wo;
            %The following is the weight adjustment from the input layer to the hidden layer
            for i=1:1:5
                dO(i)=4/(exp(I(i))+exp(-I(i)))^2;%(1*5 matrix)
            end
            segma=delta3*wo;%(1*5 matrix, row vector)
            delta2 = dO.*segma;
            d_wi = delta2'*xi+alfa*(wi_1-wi_2);
            wi=wi_1+d_wi;
            wo_3=wo_2;
            wo_2=wo_1;
            wo_1=wo;%Store the adjusted weights of the output layer
            wi_3=wi_2;
            wi_2=wi_1;
            wi_1=wi;%Store the adjusted weights of the hidden layer
         Kp=K(1);Ki=K(2);Kd=K(3);
         sys=[u_k,Kp,Ki,Kd];       