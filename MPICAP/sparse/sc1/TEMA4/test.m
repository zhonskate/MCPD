k= -9999:10000:9999;
l=1
[R,RR,v] = GenerateDiag(10000,k);
l=2
tic; y = DiagonalMultCols(RR',k',10000,10000,v);toc
l=3
tic; R*v;toc
clear
