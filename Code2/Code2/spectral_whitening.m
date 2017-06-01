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
upK = 10;
proxK = 10;
setIndexi =4;
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
if isGene == 1
    data_original_tmp = load('../Data/GSE2187/GSE2187.csv');
    fid1 = fopen('../Data/GSE2187/GSE2187-title.csv');
    X = textscan(fid1, '%s', 'Delimiter','\n');
    fclose(fid1);
    a = X{1,1};
    celldata = cellstr(a);
end
% data: 500 samples, each sample is a 6x6 image 
[data_original , z_true, pi_true] = dataGen(num,sigma_n);

if isGene == 1
    load('rank500.mat')
    data_original = data_original_tmp(allrank,:);
end
[dim,num] = size(data_original);
%data_original = data_original./repmat(max(abs(data_original'))',[1,num]);
d = dim;
data_original = double(data_original);
%data_original = log(data_original+ones(d,num).*0.0001);
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
tic

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
 lambda_list
for i = 1:i_tmp-1
    real_lambda = roots([4+lambda_list(i)^2  -1*lambda_list(i)^2-4 1]);
    real_lambda(1)
    lambda_list(i) = (6*real_lambda(1)^2- 6*real_lambda(1) +1)/((-1)*real_lambda(1)^2 +real_lambda(1));
end
 lambda_list
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

theta_lst = repmat((ones(indexi,1)./(scale))', [d,1]).*(pinv(W)'*theta_lst_tmp); %
time = toc;
theta_tmp = theta_lst;





if dimension_reduction == 1
    theta_lst = (PC*theta_tmp);
    theta_lst_backup = theta_lst;
    %theta_lst = theta_lst+ repmat(M1_original,[1,indexi]);
else
    theta_lst = theta_tmp;
end
%theta_lst = theta_lst(:,[2 1 3 4]);

if shown_reduction == 1
    k = min(1000,dim);
    [maxi index_theta] = max(abs(theta_lst));
    for i = 1:indexi
        theta_lst(:,i) = theta_lst(:,i).*sign(theta_lst(index_theta(i),i));
    end
    [amplitude,indexlist] = sort(sum(abs(theta_lst)'),'descend');
    theta_lst_tmp2 = theta_lst(indexlist(1:k),:);
    weight = [];
    for i = 1:k
        weight = [weight 1^(indexi+1-i)]; 
    end
    weight = [-k:2:-1  2:2:k];
    [ampli, indexlst] = sort(theta_lst_tmp2'*(weight'));
    %theta_lst = (PC(:,start:100)*PC2(:,indexi+1:end));
    imagesc(theta_lst_tmp2(:,indexlst));
    colorbar;
    
%cgo = clustergram(theta_lst_tmp2,'Standardize','Row','Cluster','col');
end

if cleaner == 1
    for i = 2:indexi
        theta_lst(:,i) = theta_lst(:,i) - theta_lst(:,1:i-1)*(theta_lst(:,i)'*theta_lst(:,1:i-1))';
    end
    %theta_lst(:,1)=theta_lst(:,1)./norm(theta_lst(:,1));
end


if showImage == 1
    figure(8)
    hold on 
    if isImage == 1

    [maxi, index] = max(abs(theta_lst(:,1)));
    scale = 1;
    if theta_lst(index,1) < 0
        scale = -1;
    end
        theta_lst(:,1) = scale.*theta_lst(:,1);
        tmp = vec2mat((theta_lst(:,1)),img_len)'./max(abs(theta_lst(:,1)));
        as(:,:,1) = tmp(:,1:img_wid);
        if image3color == 1
        as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
        as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
        end
        subplot(1,4,1)
        imshow(as)
    end

if isImage == 1
    [maxi, index] = max(abs(theta_lst(:,2)));
    scale = 1;
    if theta_lst(index,2) < 0
        scale = -1;
    end
    theta_lst(:,2) = scale.*theta_lst(:,2);
    tmp = vec2mat((theta_lst(:,2)),img_len)'./max(abs(theta_lst(:,2)));
as(:,:,1) = tmp(:,1:img_wid);
    if image3color == 1
    as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
    as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
    end
    subplot(1,4,2)
    imshow(as)
end
if isImage == 1
    [maxi, index] = max(abs(theta_lst(:,3)));
    scale = 1;
    if theta_lst(index,3) < 0
        scale = -1;
    end
    theta_lst(:,3) = scale.*theta_lst(:,3);
    tmp = vec2mat((theta_lst(:,3)),img_len)'./max(abs(theta_lst(:,3)));
    as(:,:,1) = tmp(:,1:img_wid);
        if image3color == 1
        as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
        as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
        end
        subplot(1,4,3)
        imshow(as)
    end
    if isImage == 1
        [maxi, index] = max(abs(theta_lst(:,4)));
        scale = 1;
        if theta_lst(index,4) < 0
            scale = -1;
        end
        theta_lst(:,4) = scale.*theta_lst(:,4);
        tmp = vec2mat((theta_lst(:,4)),img_len)'./max(abs(theta_lst(:,4)));
        as(:,:,1) = tmp(:,1:img_wid);
        if image3color == 1
        as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
        as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
        end
        subplot(1,4,4)
        imshow(as)
    end
    hold off
end

%}

% Decision on s_i
for i = 1:indexi
    [maxi, index] = max(abs(theta_lst(:,i)));
    if theta_lst(index,i) < 0
        theta_lst(:,i) = (-1).*theta_lst(:,i);
    end
end


%extract = (theta_lst'*theta_lst)\theta_lst'*data_original;
extract = zeros(indexi,num);
for i = 1:num
    extract(:,i) = data_original(:,i)'*theta_lst; 
    %extract(:,i) = data(:,i)'*theta_lst_tmp;
    
end




if shown_reduction == 1   
    tmp = extract(:,[1:47, 96:119, 214:237, 446:506, 528:587, 48:71, 297:320, 398:421, 507:527, 72:95, 120:143, 238:272, 321:344, 144:174, 190:213 273:296 345:397, 422:445, 175:189]);
    test = [ mean(tmp(:,1:216)')' mean(tmp(:,217:309)')' mean(tmp(:,310:416)')' mean(tmp(:,417 :572)')' mean(tmp(:,572:end)')' ];
    %extract = extract./repmat(max(abs(extract)')',[1,num]);
    [maxi index] = sort(tmp(:,216)'*tmp(:,1:215));
    tmp2 = [tmp(:,index) tmp(:,216)];
    [maxi index] = sort(tmp(:,309)'*tmp(:,217:308));
    tmp2 = [tmp2 tmp(:,index+216*ones(1,92)) tmp(:,309)];
    [maxi index] = sort(tmp(:,416)'*tmp(:,310:415));
    tmp2 = [tmp2 tmp(:,index+309*ones(1,106)) tmp(:,416)];
    [maxi index] = sort(tmp(:,572)'*tmp(:,417:571));
    tmp2 = [tmp2 tmp(:,index+416*ones(1,155)) tmp(:,572)];
    tmp2 = [tmp2 tmp(:,573:end)];
    %tmp2 = tmp2./repmat(max(abs(tmp2)')',[1,num]);
    tmp3 = [tmp2(:,1:216) repmat(mean(tmp2(:,1:216)')',[1,30]) tmp2(:,217:309) repmat(mean(tmp2(:,217:309)')',[1,30]) tmp2(:,310:416) repmat(mean(tmp2(:,310:416)')',[1,30]) tmp2(:,417:572) repmat(mean(tmp2(:,417:572)')',[1,30]) tmp2(:,573:end) repmat(mean(tmp2(:,573:end)')',[1,30])  ];
    tmp4 = [tmp(:,1:216) repmat(mean(tmp(:,1:216)')',[1,30]) tmp(:,217:309) repmat(mean(tmp(:,217:309)')',[1,30]) tmp(:,310:416) repmat(mean(tmp(:,310:416)')',[1,30]) tmp(:,417:572) repmat(mean(tmp(:,417:572)')',[1,30]) tmp(:,573:end) repmat(mean(tmp(:,573:end)')',[1,30])  ] ;
    %data_part = data_original(:,[1:47, 96:119, 214:237, 446:506, 528:587, 48:71, 297:320, 398:421, 507:527, 72:95, 120:143, 238:272, 321:344, 144:174, 190:213 273:296 345:397, 422:445, 175:189]);
    cgo = clustergram(extract,'ColumnPdist', 'euclidean','Standardize','row','Cluster','all','ColumnLabels',celldata);
    cgo = clustergram(tmp3(:,[1:end]),'ColumnPdist', 'euclidean','Standardize','row','Cluster','col');

end


%cgo = clustergram(test,'ColumnPdist', 'euclidean','Standardize','row','Cluster','col')
 

B = sort(extract(:,1:num)');
[maxi index] = max(B(round(num*0.2):num - round(num*0.2),:)-B(round(num*0.2)-1:num - round(num*0.2)-1,:));
boarder = zeros(indexi,2);
%boarder_index = zeros(1,4);
for i = 1:indexi
    %[maxi ind] = max([B(index(i)+18,i)-B(index(i)+17,i) B(index(i)+19,i)-B(index(i)+18,i)]);
    boarder(i,:) = [ B(index(i)+round(num*0.2)-2,i) B(index(i)+round(num*0.2)-1,i) ];
    %boarder_index(i) = index(i)+ind+18;
    %boarder(i,:) = [ B(50,i) B(51,i) ];
    if boarder(i,1) < 0
        boarder(i,1) = boarder(i,2)./4;
    end
end

z = zeros(indexi,num);
for i = 1:num
    %z(extract(:,i) < boarder(:,1) |extract(:,i) == boarder(:,1),i) = 0;
    z(extract(:,i) > boarder(:,2) |extract(:,i) == boarder(:,2),i) = 1;
     
    %tmpi = abs(abs(repmat(extract(:,i),[1,2])./boarder - ones(num-indexi,2)));
    
    %[mi, ind] = min(tmpi');
    %z(:,i) = ind' - ones(num-indexi,1);
    %z(tmpi(:,2) > 0.4 ,i) = 0;   %0.19 31wrong , z accurate = 99%
    %z(extract(:,i) < boarder(:,1) & extract(:,i) < boarder(:,2) >0,i) = 0;
    %z(extract(:,i) > boarder(:,2) & extract(:,i) > boarder(:,2) >0,i) = 1;
     
end
%{
figure(5)


A = (PC*PC2(:,indexi+1:end))';
z(:,3) = [1 0 0 1];
%A = data_original(:,[2])';
%A = X_hat';
A = (theta_lst*z([4 3 2 1],1:4))';
set(0,'DefaultAxesFontSize', 15)
figure(6)
tmp = vec2mat(abs(A(1,:)'),img_len)'./max(abs(A(1,:)));
as(:,:,1) = tmp(:,1:img_wid);
if image3color == 1
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(1,4,1)
imshow(as)
title('0 1 1 1')

tmp = vec2mat(abs(A(2,:)'),img_len)'./max(abs(A(2,:)));
as(:,:,1) = tmp(:,1:img_wid);
if image3color == 1
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(1,4,2)
imshow(as)
title('0 0 1 1')


tmp = vec2mat(abs(A(3,:)'),img_len)'./max(abs(A(3,:)));
as(:,:,1) = tmp(:,1:img_wid);
if image3color == 1
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(1,4,3)
imshow(as)
title('1 0 0 1')


tmp = vec2mat(abs(A(4,:)'),img_len)'./max(abs(A(4,:)));
as(:,:,1) = tmp(:,1:img_wid);
if image3color == 1
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(1,4,4)
imshow(as)
title('1 1 0 0')

%}

%{
figure(6)
weight = [];
for i = 1:indexi
   weight = [weight 2^(indexi+1-i)]; 
end
[ampli, indexlst] = sort(z'*(weight'));
%theta_lst = (PC(:,start:100)*PC2(:,indexi+1:end));
imshow(z(:,indexlst))
%}

%z = (A*A' - vari.*eye(4))\z*data';
%A = (z*z'-0.25.*eye(4))\z*data';

%A = (PC*PC2(:,indexi+1:end))';
%A = data_original(:,1:4)';
%{
index = [4 42 12 8]
z(:,12) = [1 1 1 1]';
A = ([theta_lst_scale]*z(:,index))';
set(0,'DefaultAxesFontSize', 18)
figure(6)
tmp = vec2mat(abs(A(1,:)'),img_len)'./max(abs(A(1,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,1)
imshow(as)
title('0 0 1 1')

tmp = vec2mat(abs(A(2,:)'),img_len)'./max(abs(A(2,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,2)
imshow(as)
title('1 1 0 1')


tmp = vec2mat(abs(A(3,:)'),img_len)'./max(abs(A(3,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,3)
imshow(as)
title('1 1 1 1')


tmp = vec2mat(abs(A(4,:)'),img_len)'./max(abs(A(4,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,4)
imshow(as)
title('1 0 1 0')

A = data_original(:,index)';

tmp = vec2mat(abs(A(1,:)'),img_len)'./max(abs(A(1,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,5)
imshow(as)

tmp = vec2mat(abs(A(2,:)'),img_len)'./max(abs(A(2,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,6)
imshow(as)


tmp = vec2mat(abs(A(3,:)'),img_len)'./max(abs(A(3,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,7)
imshow(as)


tmp = vec2mat(abs(A(4,:)'),img_len)'./max(abs(A(4,:)));
as(:,:,1) = tmp(:,1:img_wid);
if img_len ~= 6
as(:,:,2) = tmp(:,img_wid+1:img_wid*2);
as(:,:,3) = tmp(:,img_wid*2+1:img_wid*3);
end
subplot(2,4,8)
imshow(as)





figure(7)
figureHandle = gcf;
set(findall(figureHandle,'type','text'),'fontSize',14,'fontWeight','bold')
hold on;
plot(sort(extract(1,:)),'g')
plot(sort(extract(2,:)),'b')
plot(sort(extract(3,:)),'black')
plot(sort(extract(4,:)),'r')
plot(boarder_index(1),B(boarder_index(1),1),'gx')
plot(boarder_index(2),B(boarder_index(2),2),'bx')
plot(boarder_index(3),B(boarder_index(3),3),'blackx')
plot(boarder_index(4),B(boarder_index(4),4),'rx')
leg1 = legend('v_1', 'v_2', 'v_3', 'v_4');
set(leg1,'location','NorthWest')
xlabel('sorted number')
ylabel('projection value')
%}
%{
likelihood = 0;
for i = 1: num
     X_hat = theta_lst*z(:,i);
     norm_x_hat = sqrt(sum(X_hat.^2));
     if norm_x_hat ~= 0
        X_hat = X_hat./repmat(norm_x_hat,[dim,1]);
     end
     likelihood = likelihood + log (mvnpdf((data_original(:,i)./sqrt(sum(data_original(:,i).^2)))',X_hat' , vari.*eye(dim)));
end
likelihood

%}


fig1 = [0 1 0 0 0 0;
        1 1 1 0 0 0;
        0 1 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
    
fig2 = [0 0 0 1 1 1;
        0 0 0 1 0 1;
        0 0 0 1 1 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;];
    
fig3 = [0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        1 0 0 0 0 0;
        1 1 0 0 0 0;
        1 1 1 0 0 0;];
    
fig4 = [0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 0 0;
        0 0 0 1 1 1;
        0 0 0 0 1 0;
        0 0 0 0 1 0;];
fig = [fig1(:)';fig2(:)';fig3(:)';fig4(:)'];

%{
true_likelihood = 0;
params(2) = sigma_n;

for i = 1: num
     X_hat = fig'*z_true(:,i);
     %norm_x_hat = sqrt(sum(X_hat.^2));
     %if norm_x_hat ~= 0
     %   X_hat = X_hat./repmat(norm_x_hat,[dim,1]);
     %end
     for j = 1:d
        params(1) = data_original(j,i);%/sqrt(sum(data_original(:,i).^2));
        true_likelihood = true_likelihood + normlike(params,X_hat(j));
        %log (sum(normpdf((data_original(:,i)./sqrt(sum(data_original(:,i).^2)))',X_hat',vari)));
     end
end

true_pi = [0.5 0.5 0.5 0.5];
true_numpi = sum(z_true');
%true_likelihood = true_likelihood - log(true_pi)*true_numpi' - log(ones(1,indexi) - true_pi)*(ones(1,indexi).*num-true_numpi)';

%}



theta_lst_scale = theta_lst;
%scale = zeros(1,indexi);
pi = sum(z');
pi_tmp = pi_vector;
pi_vector = zeros(1,indexi);
for i= 1:indexi
    if pi(i)/num > 0.5
        pi_vector(i) = pi_tmp(1,i);
    else
        pi_vector(i) = pi_tmp(2,i);
    end
end

%pi_vector = zeros(1,indexi);
%pi_vector = pi./num;

%{
scale = power(abs(lambda_list'./(-6.*(pi_vector'.^4)+12.*(pi_vector'.^3) -7.*(pi_vector'.^2)+ 1.*(pi_vector'))),0.25);
for i = 1:indexi
    extract_positive = sort(extract(i,z(i,:)>0));
    num_extract = length(extract_positive);
    %scale(i) = mean(extract_positive(1:end));
    
    %[maxi index] = sort(theta_lst_scale(:,i));
    %[junk max_index] = max(maxi(d/2+1:end)./maxi(d/2:end-1));
    %scale(i) = 1/mean(maxi(d/2+max_index:end));
    for j = 1:d
        if theta_lst(j,i) > 0.1
            theta_lst_scale(j,i) = theta_lst(j,i).*scale(i);
        end
        if theta_lst(j,i) < 0.1
            theta_lst_scale(j,i) = 0;
        end
    end
    %theta_lst_scale(:,i) = theta_lst(:,i).*scale(i);
    %theta_lst_scale(index(d/2+max_index:end),i) = theta_lst(index(d/2+max_index:end),i).*scale(i);
    %theta_lst_scale(index(1:d/2+max_index-1),i) = theta_lst(index(1:d/2+max_index-1),i)./100;
end
%}


theta_lst_scale = theta_lst;
%% Generate test data
num_test = 100;
[data_test , z_true, pi_true] = dataGen(num_test,sigma_n);
extract_test = theta_lst_scale'*data_test;

B_test = sort(extract_test(:,1:num_test)');
[maxi index] = max(B_test(round(num_test*0.2):num_test - round(num_test*0.2),:)-B_test(round(num_test*0.2)-1:num_test - round(num_test*0.2)-1,:));
boarder_test = zeros(K,2);
%boarder_index = zeros(1,4);
for i = 1:indexi
    %[maxi ind] = max([B(index(i)+18,i)-B(index(i)+17,i) B(index(i)+19,i)-B(index(i)+18,i)]);
    boarder_test(i,:) = [ B_test(index(i)+round(num_test*0.2)-2,i) B_test(index(i)+round(num_test*0.2)-1,i) ];
    %boarder_index(i) = index(i)+ind+18;
    %boarder(i,:) = [ B(50,i) B(51,i) ];
    if boarder_test(i,1) < 0
        boarder_test(i,1) = boarder_test(i,2)./4;
    end
end

z_test = zeros(K,num_test);
for i = 1:num_test
    z_test(extract_test(:,i) > boarder_test(:,2) |extract_test(:,i) == boarder_test(:,2),i) = 1;

end


%% Evaluation
likelihood = 0;

params(2) = sqrt(vari);
for i = 1: num_test
     X_hat = theta_lst_scale*z_test(:,i);
     norm(X_hat-data_test(:,i));
     for j = 1:d
        params(1) = X_hat(j);
        %/sqrt(sum(data_original(:,i).^2));
        likelihood = likelihood + normlike(params,data_test(j,i));
        %log (sum(normpdf((data_original(:,i)./sqrt(sum(data_original(:,i).^2)))',X_hat',vari)));
     end
end
%{
for i = 1:indexi
   %f = [2 -2 -1 1 ].*scale(i)^4;
   f = [-6 12 -7 1 ].*(scale(i)^4);
   f = [f (-1).*lambda_list(i)];
   root = abs(roots(f));
   [tmpi, min_index]= min(abs(root - ones(indexi,1)*0.5));
   pi_vector(i) = root(min_index);
end

pi_vector = pi./num;
%}
%pi_true
%pi_vector
pi = sum(z_test');
likelihood = likelihood - log(pi_vector)*pi' - log(ones(1,indexi) - pi_vector)*(ones(1,indexi).*num_test-pi)';
likelihood = real(likelihood)
[maxi index] = max(fig*theta_lst_scale);
fig= fig';

norm_diff = norm(fig(:,index)- theta_lst_scale);

%{
    likelihood = 0;
    for i = 1:num
     likelihood = likelihood + log (sum(normpdf(data_original(:,i)',X(:,i)',vari_all)));
    end
    likelihood
%}

%save('data_tmp','data_original')
%save('ztrue_tmp','z_true')