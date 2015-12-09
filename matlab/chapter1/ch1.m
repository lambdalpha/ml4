% chapter 1 
m = [0 1]'; S = eye(2);
x1=[0.2 1.3]'; x2=[2.2 -1.3]';
pg1 = comp_gauss_dens_val(m, S, x1)
pg2 = comp_gauss_dens_val(m, S, x2)


% plot Gaussian
S = {};
N = 500;
m = [0 0]';
S{1} = [1 0 ; 0 1];
S{2} = [0.2 0 ; 0 0.2];
S{3} = [2 0; 0 2];
S{4} = [0.2 0; 0 2];
S{5} = [2 0; 0 .2];
S{6} = [1 0.5; 0.5 1];
S{7} = [0.3 0.5; 0.5 2];
S{8} = [0.3 -0.5; -0.5 2];

figure
for i = 1:8
   Sigma = S{i};
   data = gen_gaussian(m, Sigma, N);
   subplot(3, 3,i);
   plot(data(:, 1), data(:, 2), '.');
   axis equal
   axis([-7 7 -7 7])
   %figure(i), plot(data(:, 1), data(:, 2), '.');
   %figure(i), axis equal
   %figure(i), axis([-7 7 -7 7])
   
end



