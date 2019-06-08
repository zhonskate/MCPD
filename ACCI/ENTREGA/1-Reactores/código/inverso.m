function [y] = inverso(L11,L21,L22,M11,M12,x)
    val1 = M11.*x;
    val2 = M12.*x;
    val3 = val1 + val2;
    val4 = L11\val3;
    val5 = -L21.*val4;
    y = L22\val5;
end