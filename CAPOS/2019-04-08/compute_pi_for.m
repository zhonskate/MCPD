% basic monte carlo method to compute pi. This is the sequential
% version that uses a simple loop. input variable N is the number of
% points we will use to compute pi. Large value of N will result in
% a better approximation of pi,

function piresult = compute_pi_for(N)
tic;
Ns=rand(N/numlabs);
spmd
    xcoord=rand(Ns,1);
    ycoord=rand(Ns,1);
    total_inside=0;

    for i=1:N
        if (xcoord(i)^2 + ycoord(i)^2 <= 1)
            total_inside=total_inside+1;
        end
    end
end

aux=0;
for j=1:4
    aux = aux + total_inside{j};
end

piresult = 4*aux/N;
toc
fprintf('The computed value of pi is %8.7f.\n',piresult);
end