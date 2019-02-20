function [y] = DenseMatVecRow(A,x)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    [n,m] = size(A);
    y = zeros(n,1);
    for i = 1:n
        for j = 1:m
            y(i) = y(i) + A(i,j)*x(j);
        end
    end
end

