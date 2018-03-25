dataDir = '../../ConvexNMF/Experiment/Real/Mnist1k/'

%Z0 = load([dataDir 'Z0']);
%W0 = load([dataDir 'W0']);
R = load([dataDir 'R']);
[N,D] = size(R);
%[N,K] = size(Z0);
%[K,D] = size(W0);

fp_rmsenoisy = fopen([dataDir 'RMSEnoise-vs-K/BP-Means'],'w');
fprintf(fp_rmsenoisy, 'K RMSE\n');
%fp_rmse = fopen([dataDir 'RMSE-vs-K/BP-Means'],'w');
%fprintf(fp_rmse, 'K RMSE\n');
%fp_hamming = fopen([dataDir 'Hamming-vs-K/BP-Means'],'w');
%fprintf(fp_hamming, 'K Hamming-Err\n');
for K = 5:5:50
	[Z,A,pics]=find_picture_features( [dataDir 'R'] ,'../data/tabletop','jpg','Nstarts',100,'Kinput',K);
	%save( ['Z.syn1.K' num2str(K)], 'Z',  '-ascii');
	%save( ['W.syn1.K' num2str(K)], 'A', '-ascii');
	
	['K=' num2str(K)]
	rmse_noisy = norm(R-Z*A,'fro')/sqrt(N*D);
	fprintf(fp_rmsenoisy, '%d %g\n' , K,rmse_noisy);
	%rmse = norm(Z0*W0-Z*A,'fro')/sqrt(N*D);
	%fprintf(fp_rmse, '%d %g\n' , K, rmse);
	%hamErr = hammingErr(Z,Z0);
	%fprintf(fp_hamming, '%d %g\n' , K, hamErr);
end
fclose(fp_rmsenoisy);
%fclose(fp_rmse);
%fclose(fp_hamming);
