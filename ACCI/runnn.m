load biblis/biblis2.mat
x=ones(219,1);
cd 1-Reactores/
avg = 0;

for i=1:10
    tic;y=myproduct(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[myproduct biblis2] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[myproduct biblis2] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=prodNucleCgs(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[CGS biblis2] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[CGS biblis2] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=inverso(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[inverso biblis2] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[inverso biblis2] AVERAGE %d\n---------------\n', avg)


cd ..
load biblis/biblis5.mat
x=ones(1095,1);
cd 1-Reactores/

avg = 0;
for i=1:10
    tic;y=myproduct(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[myproduct biblis5] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[myproduct biblis5] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=prodNucleCgs(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[CGS biblis5] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[CGS biblis5] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=inverso(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[inverso biblis5] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[inverso biblis5] AVERAGE %d\n---------------\n', avg)
cd ..



matrixes_structured_10cm
x=ones(4040,1);
cd 1-Reactores/

avg = 0;
for i=1:10
    tic;y=myproduct(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[myproduct 10cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[myproduct 10cm] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=prodNucleCgs(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[CGS 10cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[CGS 10cm] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=inverso(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[inverso 10cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[inverso 10cm] AVERAGE %d\n---------------\n', avg)
cd ..


cd biblis/
matrixes_structured_5cm
cd ..
x=ones(23176,1);
cd 1-Reactores/

avg = 0;
for i=1:10
    tic;y=myproduct(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[myproduct 5cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[myproduct 5cm] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=prodNucleCgs(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[CGS 5cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[CGS 5cm] AVERAGE %d\n---------------\n', avg)
cd ..




cd biblis/
matrixes_unstructured_11cm
cd ..
x=ones(39945,1);
cd 1-Reactores/

avg = 0;
for i=1:10
    tic;y=myproduct(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[myproduct 11cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[myproduct 11cm] AVERAGE %d\n---------------\n', avg)

avg = 0;
for i=1:10
    tic;y=prodNucleCgs(L11,L21,L22,M11,M12,x);k = toc;
    fprintf('[CGS 11cm] %i \t %d\n', i, k)
    avg = avg +k;
end
avg = avg/10;
fprintf('---------------\n[CGS 11cm] AVERAGE %d\n---------------\n', avg)
cd ..