function [y] = myproduct(L11,L21,L22,M11,M12,x)
    if size(x) < 1100
        aux1=M11.*x;
        aux2=L21.*x;
        M2=ichol(L22);
        aux3=cgs(L22,aux2,1.e-7,100,M2,M2');
        %aux3=cgs(L22,aux2,1.e-7,100);
        aux4=aux1+M12.*aux3;
        M1=ichol(L11);
        y=cgs(L11,aux4,1.e-7,100,M1,M1');
        %y=cgs(L11,aux4,1.e-7,100);
    else
        aux1=M11.*x;
        aux2=L21.*x;
        %M2=ichol(L22);
        %aux3=cgs(L22,aux2,1.e-7,100,M2,M2');
        aux3=cgs(L22,diag(aux2),1.e-7,100);
        aux4=diag(aux1)+diag(M12.*aux3);
        %M1=ichol(L11);
        %y=cgs(L11,aux4,1.e-7,100,M1,M1');
        y=cgs(L11,aux4,1.e-7,100);
    end
        
end
