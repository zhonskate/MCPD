function [x,xml] = zeroForcing(H,b,alpha,L)
    [Q,R]=qr(H);
    c = Q'*b;
    x = R\c;
    A = zeros(1,L);
    for i = 1:L
        A(1,i) = alpha + (i-1);
    end
    [sx,k] = size(x);
    xml = zeros(sx,1);
    for i = 1:sx
        value = -inf;
        diff = inf;
        for j = 1:L
            if abs(x(i)-A(j))<diff
                diff = abs(x(i)-A(j));
                value = A(j);
            end
        end
        xml(i) = value;
    end
end

