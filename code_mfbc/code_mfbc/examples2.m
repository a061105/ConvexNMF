%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  *** Examples demonstrating the use of the code ***
%  
%
%   (C) Martin Slawski, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('.'));
filename ='Yeast' ;
% mkdir(strcat(filename,'/Hamming-vs-K'));
% mkdir(strcat(filename,'/RMSE-vs-K'));
mkdir(strcat(filename,'/RMSEnoise-vs-K'));
load(strcat('../../Experiment/Real/',filename,'/R'));
% load(strcat('../../Experiment/Real/',filename,'/W0'));
% load(strcat('../../Experiment/Real/',filename,'/Z0'));
%so now we have R,W0,Z0
datevary=datestr(datetime);
fprintf('handling %s \n',filename);
%open specific file to write specific data
% file1=fopen(strcat(filename,'/Hamming-vs-K/MF-Binary'),'w');
% file2=fopen(strcat(filename,'/RMSE-vs-K/MF-Binary'),'w');
file3=fopen(strcat(filename,'/RMSEnoise-vs-K/MF-Binary'),'w');
% fprintf(file1,'This is %s of hamming and the experiment started at %s\n',filename,datevary);
% fprintf(file2,'This is %s of RMSE and the experiment started at %s\n',filename,datevary);
fprintf(file3,'This is %s of RMSEnoise and the experiment started at %s\n',filename,datevary);
pics=R;
[sn,sd]=size(pics);
%start running 
opt0 = opt_Integerfac_findvert('nonnegative', false, 'affine', false);
for set_k = 5:5:50
	tic;
	fprintf('Now set_k=%d\n',set_k);
	[That, Ahat, status] = Integerfac_findvert_cpp(pics,set_k, [0 1], opt0);
	% e_time=toc;	
	% [~,h_err] = best_fit(Z0,That);
	% rmse_err  =  norm(Z0*W0-That*Ahat,'fro')/(sqrt(sn*sd));
	rmse_n_err = norm(pics-That*Ahat,'fro')/(sqrt(sn*sd)); 
	% fprintf(file1,'%d %g\n',set_k,h_err);
	% fprintf(file2,'%d %g\n',set_k,rmse_err);
	fprintf(file3,'%d %g\n',set_k,rmse_n_err);
end
%%% *** End of file *** %%%%%%%%%%%%%%%%%%

