x0(1)=50;
x0(2)=23;
x0(3)=16;
x0(4)=7;
x0=x0(:)
for i = 1:10
    M=difjac('f_sistema',x0);
    f = f_sistema(x0)
    s=M\(-f);
    x0=x0+s
end
