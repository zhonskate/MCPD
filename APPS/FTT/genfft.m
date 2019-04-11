function [y] = genfft(x)

    z=zeros(size(x)); 
    y=zeros(size(x));
    n = length(x);
    v = factor(n);
    if v == 1
        y = matfour(n)*x;
    else
        w = exp(-2i*pi/n);
        p = v(1);
        m =n/p;
        
        vecw = zeros(m,1);
        for i =1:m
                vecw(i)=w^(i-1);
        end
        omega = diag(vecw);
        for j=0:p-1
            z(j*m+1:(j+1)*m)= omega(j)*genfft(x(j+1:p:n));
        end
        
        for j=0:m-1
            y(j+1:m:n)= genfft(z(j+1:m:n));
        end 
    end
end

