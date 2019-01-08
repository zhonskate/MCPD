function [valor_min]=min_potencia(A)

[rows,cols] = size(A);
[Q,R]=qr(A);
AA = Q*R*A';
valor_max = valor_potencia(AA)
B = AA - valor_max*eye(rows);
valor_max_B = valor_potencia(B);
valor_min = -valor_max_B + valor_max
