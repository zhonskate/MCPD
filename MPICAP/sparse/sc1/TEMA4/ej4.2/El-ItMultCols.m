function [y] = DiagonalMultRows(A,D,n,m,b)
    % Entrada A, D, b; salida y

    [mA,nA] = size(A);
    y = zeros(1,m);


    for i = 1:mA
        for j = 1:nA
            y(i) = y(i) + A(j,i)*b(D(j,i));
        end
    end

end