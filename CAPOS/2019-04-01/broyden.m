x0(1)=50;
x0(2)=23;
x0(3)=16;
x0(4)=7;
x0=x0(:);
M=difjac('f_sistema',x0);
for i = 1:10
    f = f_sistema(x0);
    s = x0-M\f;
    M = M+(f*s')/(s'*s)
    x0=s
end
