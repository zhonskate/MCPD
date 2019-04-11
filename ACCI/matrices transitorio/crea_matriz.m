function [A1,b1]= crea_matriz
b=load('B1.txt');
L11=load('L11 1.txt');
l11=sparse(L11(:,1),L11(:,2),L11(:,3));
L22=load('L22 1.txt');
l22=sparse(L22(:,1),L22(:,2),L22(:,3));
d1=load('d12 1.txt');
d2=load('d21 1.txt');
A1=[l11 sparse(diag(d1));sparse(diag(d2)) l22];
spy(A1)
