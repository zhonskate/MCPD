function [B] = Bidiag(A)
    [n,m]=size(A);
    i=1;
    j=1;
    while i <= n-1 && j <= m-1
        [tmp,P] = Anula(A(i:n,j));
        P = [eye(i-1) zeros(i-1,size(P,1)); zeros(size(P,1),i-1) P];
        A = P*A;
        [tmp,Q] = Anula(A(i,j+1:m)');
        Q = [eye(j) zeros(j,size(Q,1)); zeros(size(Q,1),j) Q];
        A = A*Q;
        i=i+1;
        j=j+1;
    end
    
    while i<=n-1
        [tmp,P] = Anula(A(i:n,m));
        P = [eye(i-1) zeros(i-1,size(P,1)); zeros(size(P,1),i-1) P];
        A = P*A;
        i=i+1;
    end
    
    while j<=m-1
        [tmp,Q] = Anula(A(n,j+1:m)');
        Q = [eye(j) zeros(j,size(Q,1)); zeros(size(Q,1),j) Q];
        A = A*Q;
        j=j+1;
    end
    
    B = A;
    
end

