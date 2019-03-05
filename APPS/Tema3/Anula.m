function [y,Q] = Anula(x)
    [n,m] = size(x);
    v = x + sign(x(1))*norm(x,2)*[1;zeros(n-1,1)];
    Q = eye(n)-(2*(v*v'))/(v'*v);
    y = Q*x;
end

