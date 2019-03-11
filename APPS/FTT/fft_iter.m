function [x] = fft_iter(x)
    n = length(x); 
    x=bit_rev_vector(x); 
    t=log2(n);
    
    for q=1:t
        L=2^q;
        r=n/L;
        m=L/2;
        omega = zeros(1,m);
        for i = 1:m
            omega(i)=exp(-2*pi*1i/L)^(i-1);
        end
        for j=1:m
            for k=0:r-1
                tau = omega(j)*x(k*L+j+m);
                x(k*L+j+m) = x(k*L+j)- tau;
                x(k*L+j) = x(k*L+j)+ tau;
            end
        end
    end
end
