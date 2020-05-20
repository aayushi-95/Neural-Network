clear all;
clc;


Input=[1 -2 0 -1; 0 -1.5 -0.5 -1.0; -1 1 0.5 -1.0]; 
y=[-1 -1 1];
Initial_weights=[1 -1 0 0]';
eta=0.7;
weights=Perceptron(Input,y,Initial_weights,eta)

num_data=20;
Input=randn(num_data,2);
y=ones(num_data,1);
y(Input(:,2)<Input(:,1))=-1;
Sample=[Input(1:5,:) y(1:5,:)]
Initial_weights=[1 -1]';
w_Perceptron=Perceptron(Input,y,Initial_weights,eta)
figure
hold on
scatter(Input(y==1,1),Input(y==1,2),'+')
scatter(Input(y==-1,1),Input(y==-1,2),'r')
syms x1 x2
f=w_Perceptron(1)*x1+w_Perceptron(2)*x2
fimplicit(f)
title({'Perceptron Learning'});
xlabel({'X_1'});
ylabel({'X_2'});
legend('data1','data2','Decision Boundry');
hold off

function [w]=Perceptron(xtrain,ytrain,w0,eta)
%x(1),x(2)....x(n) as row vector
%w as col vector
w=w0;
norm1=10e4; tol=10e-5;
while (norm1>tol)
    w1=w;
    for i=1:size(xtrain,1)
        ycomp(i)=sign(w'*xtrain(i,:)');
        r(i)=ytrain(i)-ycomp(i);
        w = w+eta*r(i)*xtrain(i,:)';
    end
    norm1=norm(w1-w);
end
    
end