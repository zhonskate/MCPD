function [valor]=valor_potencia(A)

[rows,cols] = size(A);
q = ones([rows,1]);
conv=1;
n=10;
while conv > 0.0001
y=A*q;
conv=abs(norm(q)-n);
n=norm(q);
q=y/n;
end
vec=q;
valor=n;
