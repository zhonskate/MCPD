function [y] = prodNucleCgs(L11,L21,L22,M11,M12,x)

    if size(x) < 1100
        val1=M11.*x;
        val2=L21.*x;
        val3=cgs(L22,val2,1.e-7,100);
        val3=M12.*val3;
        val1=val1+val3;
        y=cgs(L11,val1,1.e-7,100);
    else

        val1=M11.*x;
        val2=L21.*x;
        val3=cgs(L22,diag(val2),1.e-7,100);
        val3=M12.*val3;
        val1=val1+val3;
        y=cgs(L11,diag(val1),1.e-7,100);

    end

end

