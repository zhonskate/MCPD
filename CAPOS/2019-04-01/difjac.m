function f=difjac(fun,x)
h=1e-7;
n=max(size(x));
f=zeros(n,n);
fxmen=feval(fun,x);
nx=norm(x,inf);
for i=1:n
%  xmen=x;
  xmas=x;
%  xmen(i)=xmen(i)-h;
   if (xmas(i)~=0)
      xmas(i)=xmas(i)+h*nx;
     z=(feval(fun,xmas)-fxmen)/(h*nx);
   else
      xmas(i)=xmas(i)+h;
     z=(feval(fun,xmas)-fxmen)/(h);
   end   
%  z=(feval(fun,xmas)-feval(fun,xmen))/(2*h);
  f(:,i)=z;
end


