function [y] = prodNucle(L21,L22,L11,M11,M12,x)

% [y] = prodNucle(L21,L22,L11.M11,M12,x)
% Esta funcion realiza el producto matriz por vector (y=A*x)
% No se dispone de la matriz explicita
% el resultado se almacena en el vector y

val1=M11.*x;
val2=L21.*x;
val3=gmres(L22,val2,10,1.);
val3=M12.*val3;
val1=val1+val3;
y=gmres(L11,val1);


end