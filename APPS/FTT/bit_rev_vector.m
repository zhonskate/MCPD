function [ v ] = bit_rev_vector( v )
n=length(v);
for k=0:n-1
  j=bit_rev(k,n);
  if j>k
      aux=v(j+1);
      v(j+1)=v(k+1);
      v(k+1)=aux;
     %swap(v(j+1), v(k+1))
  end
end


