A = rand(1000);
B = rand(1000);

[m,n]=size(A);
[m,p]=size(B);
C = zeros(m,p);

tic
for i=1:m
    for j=1:p
        for k=1:n
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc

C = zeros(m,p);

tic
for i=1:m
    for k=1:n
        for j=1:p
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc

C = zeros(m,p);

tic
for j=1:p
    for i=1:m
        for k=1:n
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc

C = zeros(m,p);

tic
for j=1:p
    for k=1:n
        for i=1:m
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc

C = zeros(m,p);

tic
for k=1:n
    for i=1:m
        for j=1:p
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc

C = zeros(m,p);

tic
for k=1:n
    for j=1:p
        for i=1:m
            C(i,j)=C(i,j)+A(i,k)*B(k,j);
        end
    end
end
toc