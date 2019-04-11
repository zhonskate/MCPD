function A=migenmat(n,tbl)
A=zeros(n*tbl,n*tbl);
for i=1:n
    bl=rand(tbl);
    i_ini=(i-1)*tbl+1;
    i_fin=i*tbl;
    A(i_ini:i_fin,i_ini:i_fin)=bl;
end    