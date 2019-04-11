% basic monte carlo method to compute pi. This is the sequential
% version that uses a simple loop. input variable N is the number of
% points we will use to compute pi. Large value of N will result in
% a better approximation of pi. This version does not create the random
% number before hand but will generate a new singel random number during
% every iteration.

function piresult = compute_pi_spmd(N)
tic;
Ns = round(N/4);
numlabs = 4;
spmd
    xcoord = rand(Ns,1);
    ycoord = rand(Ns,1);
    total_inside=0;

    for i=1:N
        if (xcoord^2 + ycoord^2 <= 1)
            total_inside=total_inside+1;
        end
    end
    
    
end
aux=0;

piresult = 4*total_inside{1}/N;
toc
fprintf('The computed value of pi is %8.7f.\n',piresult);
end
