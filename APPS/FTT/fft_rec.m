function [y] = fft_rec(x)
    [m,n] = size(x);
    if n==1
        y=x;
    else
        m = n/2;
        w = exp(2*pi*1i/m);
        vec = zeros(1,m);
        for i = 1:m
            vec(i)=w^(i-1);
        end
        omega = diag(vec);
        zt = fft_rec(x(1:2:n));
        zb = omega*fft_rec(x(2:2:n));
        y = [zt+zb;zt-zb];
    end
end

