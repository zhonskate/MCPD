function [y] = DenseMatVecCol(A,x)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    [n,m] = size(A);
    y = zeros(n,1);
    for j = 1:m
        for i = 1:n
            y(i) = y(i) + A(i,j)*x(j);
        end
    end
end

