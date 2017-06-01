function [ data, z_true, pi_true ] = dataGen(n,sigma_n)
%Generate Data from the combination of basis image
%  For basis figure + Gaussian noise
alpha = 10;
K=4;
%% Image Basis:
fig1 = [0 1 0 0 0 0;
        1 1 1 0 0 0;
        0 1 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
%fig1 = fig1./norm(fig1);
    
fig2 = [0 0 0 1 1 1;
        0 0 0 1 0 1;
        0 0 0 1 1 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
%fig2 = fig2./norm(fig2);
    
fig3 = [0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        1 0 0 0 0 0;
        1 1 0 0 0 0;
        1 1 1 0 0 0;];
%fig3 = fig3./norm(fig3);    
fig4 = [0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 1 1 1;
        0 0 0 0 1 0;
        0 0 0 0 1 0;];
    
%fig4 = fig4./norm(fig4);
    fig = [fig1(:)';fig2(:)';fig3(:)';fig4(:)'];
%%


data = [];
z_true = [];
%pi_true = betarnd(1,alpha/K,[1,K]);
pi_true = [0.5 0.5 0.5 0.5];
for i = 1:n
    img_bin = K.*ones(1,K);
    img_rnd = rand(1,4) ; %1.4999.*rand -> good! why?
    img_bin = img_rnd < pi_true;
    z_true = [z_true img_bin'];
    % 0.5, 0.25, 0.188
    % 0.1, 0.01, 0.0003
    img_rand = fig1.*img_bin(1) + fig2.*img_bin(2) + fig3.*img_bin(3) + fig4.*img_bin(4)+sigma_n.*randn(6,6);
    data = [data img_rand(:)];
end

end

