function j=bit_rev(k,n)
t=log2(n);
j=0; m=k;
for q=0:t-1
	s=floor(m/2);  % bq  = m-2s
	j=2*j+(m-2*s);
	m=s;
end

