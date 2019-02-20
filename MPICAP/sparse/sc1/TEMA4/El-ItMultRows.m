function [y] = DiagonalMultRows(A,D,n,m,b)
    % Entrada A, D, b; salida y

    [nA,mA] = size(A);
    y = zeros(m,1);


    for i = 1:nA
        for j = 1:mA
            y(i) = y(i) + A(i,j)*b(D(i,j));
        end
    end

end