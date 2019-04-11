function piresult = compute_pi_matlab(N)
tic;
x=rand(N,1);
y=rand(N,1);
total_inside= sum(((x .* x + y .* y) < 1)); 
piresult = 4*total_inside/N;
toc
fprintf('The computed value of pi is %8.7f.\n',piresult);
end
