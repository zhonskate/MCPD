function [y] = myproduct(L11,L21,L22,M11,M12,x)
% [y] = untitled2(L11,L12,L22,M11,M12,x)
%Esta función calcula el producto de la matriz implícita A por el vector x
%la solución se almacena en y
aux1=M11.*x;
aux2=L21.*x;
M2=ichol(L22);
aux3=cgs(L22,aux2,1.e-7,100,M2,M2');
aux4=aux1+M12.*aux3;
M1=ichol(L11);
y=cgs(L11,aux4,1.e-7,100,M1,M1');

end

