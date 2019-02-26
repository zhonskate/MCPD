function [NewQ, NewR ]=QRfactor( Q, R, u, v)
    %Compute the QR factorization of Q*R + u*v': 
    
    %Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = Q'*u;
    n = size(Q, 1);
    
    %Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        [c, s, r] = GivensRotation(w(k), w(k+1));
        w(k+1) = 0.; w(k) = r;
        % Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R(k,j) - s*R(k+1,j);
            R(k+1,j) = s*R(k,j) + c*R(k+1,j);
            R(k,j) = newrow;
            newcol = c*Q(j,k) - s*Q(j,k+1);
            Q(j,k+1) = s*Q(j,k) + c*Q(j,k+1);
            Q(j,k) = newcol;
        end
    end
    % R <- R + w*v' is now upper-hessenberg:
    R(1,:) = R(1,:) + w(1)*v ;
    
    [Q1, R1] = HessenbergQR( R );
    
    % Return updated QR factorization: 
    NewQ = Q*Q1;
    NewR= R1;
    
end
