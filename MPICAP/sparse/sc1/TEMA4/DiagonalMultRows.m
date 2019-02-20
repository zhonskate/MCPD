function [y] = DiagonalMultRows(A,d,m,n,b)
    % Entrada A, d,m,n, b; salida y

    [N,n_diag] = size(A);
    y = zeros(m,1);


    for i = 1:N
        for j = 1:n_diag
            ccol = d(j) + i;
            if ccol > 0 && ccol <= n 
                y(i) = y(i) + A(i,j)*b(i);
            end
        end
    end

end