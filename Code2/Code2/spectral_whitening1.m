%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Fish Tung
%Name: Spectral Methods for Latent Variable
%Last Update: 2014/5/7
%%%%%%%%%%%%%%%%%%%%%%%%%8%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [norm_diff, out,likelihood, time] = spectral4_whitening(sigma_n,num)

% note: 01/24/2017
% line 224: theta_lst is the latent topics
% pi_list is the probability for latent topics
% line 45: data_original is your data input [dimension x num_samples]
sigma_n = 0.1;
mode = 2;
num = 500;
%start = 1;
dimension_reduction =0;
shown_reduction = 0;
isImage = 1;
showImage = 1;
isGene = 0;
%aware
proxK = 10;
%setIndexi=8;
sigma_given =0;
cleaner = 0;
image3color = 0;

%load('../Finale''s Code/vibp/data_synthesize_0.1.mat');
%true_sv = sigma_n;
%data_original = X';
%num = 100;
%sigma_n = 0.1;
img_len = 6; %240
img_wid = 6; %320
%data_original = import_file();
% data: 500 samples, each sample is a 6x6 image 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set_iter = 5:5:50;
re = 1:20;
filename ='Yeast' ;
upK = 60;
tol = 0;
wall = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mkdir(strcat(filename,'/Hamming-vs-K'));
% mkdir(strcat(filename,'/RMSE-vs-K'));
mkdir(strcat(filename,'/RMSEnoise-vs-K'));
load(strcat('../../Experiment/Real/',filename,'/R'));
% load(strcat('../../Experiment/',filename,'/W0'));
% load(strcat('../../Experiment/',filename,'/Z0'));
%so now we have R,W0,Z0
datevary=datestr(datetime);
fprintf('handling %s \n',filename);
%open specific file to write specific data
% file1=fopen(strcat(filename,'/Hamming-vs-K/Spectral'),'w');
% file2=fopen(strcat(filename,'/RMSE-vs-K/Spectral'),'w');
file3=fopen(strcat(filename,'/RMSEnoise-vs-K/Spectral'),'w');
% fprintf(file1,'This is %s of hamming and the experiment started at %s\n',filename,datevary);
% fprintf(file2,'This is %s of RMSE and the experiment started at %s\n',filename,datevary);
fprintf(file3,'This is %s of RMSEnoise and the experiment started at %s\n',filename,datevary);
pics=R;
[sn,sd]=size(pics);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data_original , z_true, pi_true] = dataGen(num,sigma_n);
data_original=pics'+wall;
[dim,num] = size(data_original);
%data_original = data_original./repmat(max(abs(data_original'))',[1,num]);
d = dim;
data_original = double(data_original);
%data_original = log(data_original+ones(d,num).*0.0001);

for iter=set_iter
    for rre = re
        tic;
        setIndexi=iter
        fprintf('This is iter %d re %d\n',iter,rre);
        M1_original = mean(data_original')';   
        M1_original_norm = M1_original./sqrt(sum(M1_original.^2));


        if dimension_reduction == 1
            kernel = data_original'-repmat(M1_original',[num, 1]);
            [PC1,score,D1,tsquare] = princomp(kernel*kernel','econ');
            %[PC1, D1] = eig(kernel*kernel','econ');
            %D = D1./num;
            %diagD = diag(D);
            diagD = D1;
            diagD_tmp = diagD;
            diagD_tmp(diagD_tmp < 0.00001) = 0.1;
            diagD_tmp(1) = 10000000;
            rateD = diagD(1:end-1)./diagD_tmp(2:end);
            [maxi, indexi] = max(rateD);

            PC = kernel'*PC1;
            PC = PC./repmat(sqrt(sum(PC.^2)),[d,1]);
            u = mean(PC(:,indexi+1:end)')';
            u = u'./norm(u);
            vari_all = mean((u*(data_original - repmat(M1_original,[1,num]))).^2);
            

            %M1_proj = data_original'*M1_original_norm;

            data = PC'*data_original;
            %[(V(:,2:100)'*(data_original - M1_original_norm*M1_proj'))' M1_proj]';

        else 
            data = data_original;% - repmat(M1_original,[1,num]);
        end

        %% Information of Data
        [d,num] = size(data);
        M1 = mean(data')';


        %% Find rank by random projection
        R = randn(d,upK); %d*upK 
        data_R = data'*R; 
        RM2R = data_R'*eye(num)*(data_R)./num;
        S1R = M1'*R;
        [PC3,score,D,tsquare] = princomp(RM2R-S1R'*S1R); %upK*upK svd
        indexi = findLargeSlope(D,setIndexi);
        K = indexi;

        %% noise 
        if sigma_given == 1
            u = 0;
            vari = sigma_n^2;%given
            vari4 = 3*(vari^2);
            m1 = vari.*M1;
        else
            [PC2,score,D2,tsquare] = princomp(data*data'./num - M1*M1','econ');
            u = mean(PC2(:,indexi+1:end)')';
            u = u'./norm(u);
            vari = mean((u*(data - repmat(M1,[1,num]))).^2);
            vari4 = mean((u*(data - repmat(M1,[1,num]))).^4)./3;
            m1 = data*((u*(data - repmat(M1,[1,num]))).^2)'./num;
        end


        %% W for whitening 
        KK = K;
        R = randn(d,KK); %d*K 
        data_R = data'*R; 
        M2R = data*eye(num)*(data_R)./num;
        S1R = M1*(M1'*R);
        U = M2R - S1R - vari.*R; % U = S2R

        [U, tmp] = eigs(data*eye(num)*data'./num - M1*M1' -vari.*eye(d),K );

        data_U = data'*U; 
        UM2U = data_U'*eye(num)*(data_U)./num;
        S1U = M1'*U;
        [A,Sigma,~] = svd(UM2U-S1U'*S1U-vari.*(U'*U)); %K*K svd
        V = A* diag(diag(Sigma).^(-0.5));
        Vplus = A* diag(diag(Sigma).^(0.5));

        W = U*V;
        B = U*Vplus;

        % check correctness
        % M2 = data*eye(num)*(data')./num;
        % S2 = M2 - M1*M1' - vari.*eye(d);
        % W'*S2*W

        %% 

        lambda_list = [];
        theta_lst = [];
        theta_lst_tmp = [];


        %indexi = 4;
        stop = 1;
        if  mode == 2
        i_tmp = 1;
        for i = 1:indexi
            [lambda,v, out] = fast_tensor3_whitening(m1,M1,lambda_list,theta_lst_tmp,data,vari,vari4,stop,u,W,KK);
            if lambda < 1
                  i_tmp = i;
                break;
            end
            lambda_list = [lambda_list lambda];
            theta_lst_tmp = [theta_lst_tmp v];
            if i == indexi
                i_tmp = i+1;
            end
        end
         
        for i = 1:i_tmp-1
            real_lambda = roots([4+lambda_list(i)^2  -1*lambda_list(i)^2-4 1]);
            lambda_list(i) = (6*real_lambda(1)^2- 6*real_lambda(1) +1)/((-1)*real_lambda(1)^2 +real_lambda(1));
        end
         
        for i = i_tmp:indexi
             [lambda,v, out] = fast_tensor4_whitening(m1,M1,lambda_list,theta_lst_tmp,data,vari,vari4,stop,u,W,KK);
            lambda_list = [lambda_list lambda];
            theta_lst_tmp = [theta_lst_tmp v];
        end

        end


        if  mode == 1  %use only 4
            for i = 1:indexi
            [lambda,v, out] = fast_tensor4_whitening(m1,M1,lambda_list,theta_lst_tmp,data,vari,vari4,stop,u,W,KK);
            lambda_list = [lambda_list lambda];
            theta_lst_tmp = [theta_lst_tmp v];
            end
        end

        if  mode == 3
        i_tmp = 1;
        for i = 1:indexi
            [lambda,v, out] = fast_tensor4_whitening(m1,M1,lambda_list,theta_lst_tmp,data,vari,vari4,stop,u,W,K);
            lambda_list = [lambda_list lambda];
            theta_lst_tmp = [theta_lst_tmp v];
        end
        theta_lst_tmp (:,lambda_list > -1) = [];
        lambda_list(lambda_list > -1) = []
        i_tmp = length(lambda_list)+1
        for i = 1:i_tmp-1
            real_lambda = real(roots([6+lambda_list(i)  -lambda_list(i)-6 1]));
            real_lambda(1) 
            lambda_list(i) = real(((-2)*real_lambda(1)+1)/sqrt((-1)*real_lambda(1)^2 +real_lambda(1)));
        end

        for i = i_tmp:indexi
            [lambda,v, out] = fast_tensor3_whitening(m1,M1,lambda_list,theta_lst_tmp,data,vari,vari4,stop,u,W,K);
            lambda_list = [lambda_list lambda];
            theta_lst_tmp = [theta_lst_tmp v];
        end
        end

        pi_list = zeros(1,indexi);
        pi_vector = zeros(2:indexi);
        for i= 1:indexi
            pi_list_tmp = roots([6 + lambda_list(i), -6 - lambda_list(i),1  ]);
            pi_vector(:,i) = pi_list_tmp;
            pi_list(i) = real(pi_list_tmp(1));
        end

        scale = (6*pi_list'.^2 - 6*pi_list' +1)./(lambda_list'.* sqrt(-pi_list.^2 + pi_list)');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        theta_lst = repmat((ones(indexi,1)./(scale))', [d,1]).*(pinv(W)'*theta_lst_tmp); %
        e_time=toc;
        fprintf('setIndexi=%d,tol=%f \n',setIndexi,tol);   
        theta_cor = abs(theta_lst);
        theta_cor(theta_cor < tol) = 0 ;
        [~,b_Z] = bestZErr(data_original',theta_cor');
        % [~,h_err] = best_fit(Z0,b_Z);
        % rmse_err  =  norm(Z0*W0-(b_Z*theta_cor'-wall),'fro')/(sqrt(sn*sd));
        rmse_n_err = norm(pics-(b_Z*theta_cor'-wall),'fro')/(sqrt(sn*sd)); 
        % fprintf(file1,'%d %g\n',setIndexi,h_err);
        % fprintf(file2,'%d %g\n',setIndexi,rmse_err);
        fprintf(file3,'%d %g\n',setIndexi,rmse_n_err);
    end
end
