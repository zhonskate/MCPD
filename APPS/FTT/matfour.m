function [ Mat ] = matfour(N )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
WN=exp(j*2*pi/N);
mat=zeros(N);
for i=1:N; vecWN(i)=WN^(i-1); end
vecWN=vecWN';
for i=1:N; Mat(:,i)=vecWN.^(i-1); end

end

