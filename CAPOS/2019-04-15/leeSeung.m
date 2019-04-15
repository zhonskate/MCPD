function [W,H] = leeSeung(A,K)

[rows,cols] = size(A);

W = rand(rows,K);
H = rand(K,cols);

iter = 0;
while 1
    if (iter==100)
        break
    end
    iter = iter+1;    
    H = H.*(W'*A)./((W'*W)*H + 0.000001);
    W = W.*(A*H')./(W*(H*H')+ 0.000001);
end

