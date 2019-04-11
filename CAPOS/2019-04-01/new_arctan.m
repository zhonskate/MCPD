x=-15:0.1:15;
y=atan(x);
plot(x,y)
grid
hold on
x0=5;
pause
der=1/(1+x0^2);
xn=x0-(atan(x0)/der)
line([x0,xn],[atan(x0),atan(xn)]);
pause
x0=xn;
der=1/(1+x0^2);
xn=x0-(atan(x0)/der)
line([x0,xn],[atan(x0),atan(xn)]);