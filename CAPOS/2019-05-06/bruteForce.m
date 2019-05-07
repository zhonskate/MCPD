function [xml] = bruteForce(H,b,alpha,L)
    A = zeros(1,L);
    for i = 1:L
        A(1,i) = alpha + (i-1);
    end
    [sx,k] = size(b);
    xml = zeros(sx,1);
    test = xml;
    min = inf;
    codedtest = ones(sx,1);
    k = true;
    while k
        for i = 1:sx
            test(i) = A(1,codedtest(i));
        end
        if norm(H*test - b) < min
            xml = test;
            min = norm(H*test - b);
        end
        index = 1;
        codedtest(index) = codedtest(index) + 1;
        while codedtest(index) > L
            codedtest(index) = 1;
            index = index + 1;
            if index > sx
                k = false;
                break;
            end
            codedtest(index) = codedtest(index) + 1;
        end
    end
end