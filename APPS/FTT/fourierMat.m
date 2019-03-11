N = 16;

a = zeros(N);
w = exp(-1i*2*pi/N);

for i = 1:N
    for j = 1:N
        a(i,j)=w^((i-1)*(j-1));
    end
end