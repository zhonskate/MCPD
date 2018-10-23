function C = prodmat(A,B)
    [m,n]=size(A);
    [m,p]=size(B);
    C = zeros(m,p);
    for i=1:m
        for j=1:p
            for k=1:n
                C(i,j)=C(i,j)+A(i,k)*B(k,j);
            end
        end
    end