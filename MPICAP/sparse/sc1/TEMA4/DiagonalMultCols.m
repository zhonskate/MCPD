function [y] = DiagonalMultCols(A,d,m,n,b)
    % Entrada A, d,m,n, b; salida y

    [n_diag,N] = size(A);
    y = zeros(1,m);


    for i = 1:N
        for j = 1:n_diag
            ccol = d(j) + i;
            if ccol > 0 && ccol <= n 
                y(i) = y(i) + A(j,i)*b(i);
            end
        end
    end

end