function [s] = ValSin(A)
    tol = 0.0001;
    err = norm(A-diag(diag(A)),2);
    while err > tol
        [Q,R]=qr(A);
        A = R';
        err = norm(A-diag(diag(A)),2);
    end
    s=abs(diag(A));
    
end

