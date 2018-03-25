%partial solve: Ra=1e4*R;Za=1e4*Z0; [V2,A,obj]=CDS_D(Ra,Za,1000,1)
%Lambda = 100000;
Tau = 1;

rand('seed',4);
randn('seed',4);

%Simulate Data
N = 1000;
K = 10;
D = 1000; 
%d1 = 30;
%d2 = 30;
%p1 = 7;
%p2 = 7;
NOISE = 1e-1;
[W0,Z0,R] = simulate1( N,D,K, NOISE );
%[W0,Z0,R] = simulate2( N,d1,d2,p1,p2,K, NOISE );
%[W0,Z0,R] = simulateMixReg( N,D,K, NOISE );
%return;
%Read data from file
%data_path = '../data/tabletop.pca';
%data_path = '../data/tabletop3';
%data_path = '../data/simmulated';
%data_path = 'R'
%data_path = '../data/mnist.scale.1k';
%[y,R]=libsvmread(data_path);
%data_path = '../data/yaleFace';
%data_path = '../data/R.scene';
%data_path = '../data/R.yeast';
%data_path = '../data/R.medical';
%data_path = '../data/R.synthetic';
%data_dir = '../Experiment/Simulated/';
%data_dir = '../Experiment/tabletop/';
%data_dir = '../Experiment/Syn1/';
%data_dir = '../Experiment/Syn2/';
%data_dir = '../Experiment/Syn3/';
%data_dir = '../Experiment/Real/Mnist1k/';
%data_dir = '../Experiment/Real/YaleFace/';
%data_dir = '../Experiment/Real/Yeast/';
%K_range = 5:5:50;
K_range = 1:3:15;
%K_range = 3;
Lambda = N;

%R = load([data_dir 'R']);
%Z0 = load([data_dir 'Z0']);
%V0 = load([data_dir 'W0']);
[N,D]=size(R);
R = sparse(R);

%writeMat(R,'R');
%R2 = full(R);
%save 'R' R2 -ascii;
%save 'Z0' Z0 -ascii;
%save 'W0' W0 -ascii;
%Z0 = zeros(N,10);
%Z0 = load('../data/Z.scene');
%Z0 = load('../data/Z.yeast');
%Z0 = load('../data/Z.medical');
%W0 = inv(Z0'*Z0)*Z0'*R;

%true_error = norm(R-Z0*W0,'fro')^2
%solve 
%R_mean = ones(N,1)*mean(R);
[Z,V,c] = convexBMF( R, Lambda, Tau, Z0, K_range );


%L = length(K_range);
%% RMSE noisy
%fp = fopen([data_dir 'RMSEnoise-vs-K/LatentLasso'], 'w');
%fprintf(fp, 'K RMSE\n');
%for l = 1:L	
%		Z = bestZ{l};
%		V = bestV{l};
%		rmse = norm(R-Z*V,'fro')/sqrt(N*D);
%		fprintf(fp, '%d %g\n', K_range(l), rmse);
%end
%fclose(fp);
%
%%RMSE
%fp = fopen([data_dir 'RMSE-vs-K/LatentLasso'], 'w');
%fprintf(fp, 'K RMSE\n');
%for l = 1:L	
%		Z = bestZ{l};
%		V = bestV{l};
%		rmse = norm(Z0*V0-Z*V,'fro')/sqrt(N*D);
%		fprintf(fp, '%d %g\n', K_range(l), rmse);
%end
%fclose(fp);
%
%
%%Hamming loss
%fp = fopen([data_dir 'Hamming-vs-K/LatentLasso'], 'w');
%fprintf(fp, 'K HammingError\n');
%for l = 1:L	
%		Z = bestZ{l};
%		V = bestV{l};
%		err = hammingErr(Z,Z0);
%		fprintf(fp, '%d %g\n', K_range(l), err);
%end
%fclose(fp);


%cV = V*diag(sqrt(c));
%cZ = Z*diag(sqrt(c));
%
%%obj_true = PrimalLoss(10*R/sqr_Tau, Z0/sqr_Tau, V_true) + (Lambda/Tau)*K
%
%%find top support
[c,ind]=sort(c,'descend'); 
Z = Z(:,ind);
V = V(:,ind);
Z2 = Z(:,1:K);
V2 = V(:,1:K);
c2 = c(1:K);
cV2 = V2*diag(sqrt(c2));

%[Z0 NaN(N,1) Z2]

W = inv(Z2'*Z2)*Z2'*R;
tmp = R-Z2*W;
['loss=' num2str(sum(sum(tmp.*tmp)))]
