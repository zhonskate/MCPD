function [lu_kji] = lu_kji(A)
    [m,n]=size(A);
    for k = 1:n-1
        if A(k,k)==0
            return
        end
        for i = k+1:n
            A(i,k) = A(i,k)/A(k,k);
        end
        for j = k+1:n
            for i = k+1:n
                A(i,j)=A(i,j)-A(k,j)*A(i,k);
            end
        end
    end
    lu_kji = A;
end

