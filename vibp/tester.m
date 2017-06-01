clear
addpath('vibp_base/')
addpath('util/')
addpath('heuristics/')
%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%
re_set=1:3;
set_Kr = 2:10;
alpha_base = 0.5;
filename ='tabletop' ;
%%%%%%%%%%%%%%%%%%%%%
load( 'data.mat' );
mkdir(strcat(filename,'/Hamming-vs-K'));
mkdir(strcat(filename,'/RMSE-vs-K'));
mkdir(strcat(filename,'/RMSEnoise-vs-K'));
load(strcat('../Experiment/',filename,'/R'));
load(strcat('../Experiment/',filename,'/W0'));
load(strcat('../Experiment/',filename,'/Z0'));
[set_N,~] =size(R);
%so now we have R,W0,Z0
datevary=datestr(datetime);
fprintf('handling %s \n',filename);
%open specific file to write specific data
file1=fopen(strcat(filename,'/Hamming-vs-K/Variational'),'w');
file2=fopen(strcat(filename,'/RMSE-vs-K/Variational'),'w');
file3=fopen(strcat(filename,'/RMSEnoise-vs-K/Variational'),'w');
fprintf(file1,'This is %s of hamming and the experiment started at %s\n',filename,datevary);
fprintf(file2,'This is %s of RMSE and the experiment started at %s\n',filename,datevary);
fprintf(file3,'This is %s of RMSEnoise and the experiment started at %s\n',filename,datevary);
[sn,sd]=size(R);
for s_alpha = 1:5    
      for re = re_set 
        for set_K = set_Kr       
            X=R;
            N=set_N;    
            alpha=alpha_base*pow2(s_alpha-1);             
            param_set.K                       = set_K;    
            fprintf('Data:%s,alpha=%f,re=%d,setK=%d,N=%d\n',filename,alpha,re,param_set.K,N);   
            %R_yal 165,R_mn 1000 1211  978  
            % -------------------------- %
            %       Set Parameters       %
            % -------------------------- %
            % set basic parameters of the data
            param_set.alpha = alpha;                      % IBP concentration parameter
            param_set.sigma_n = sigma_n;                  % noise variance
            param_set.sigma_a = sigma_a;                  % feature variance

            % set basic parameters of the algorithm
                   % truncation level
            param_set.restart_count           = 1;        % number of random restarts
            param_set.vibp_iter_count         = 2;        % number of inner optimisation iterations to perform between heuristic search moves
            param_set.use_tempering           = true;     % whether variances should be tempered (recommended)
            param_set.compute_intermediate_lb = false;    % should lb be computed after each parameter update (for logging purposes)
            param_set.stopping_thresh         = 0.1;      % multiplicative difference in lower bounds to stop optimising

            % set output parameters
            param_set.show_final_plots        = false;    % should final Z/reconstructions be shown at end?

            % set held out data: test_mask( i,j ) = 1 indicates data is to be used for training.
            % this just demonstrates how to set up a mask to treat part of the data as unobserved.
            param_set.test_mask = ones( size( X ) );
            % test_n_start1_ind = N - floor( N / 2 );
            % test_n_end1_ind = N - ceil( N / 3 );
            % test_n_start2_ind = N - floor( N / 3 );
            % test_n_end2_ind = N - ceil( N / 6 );
            % test_n_start3_ind = N - floor( N / 6 );
            % test_n_end3_ind = N;
            % param_set.test_mask( test_n_start1_ind:test_n_end1_ind , 1:3:end ) = 0;
            % param_set.test_mask( test_n_start2_ind:test_n_end2_ind , 2:3:end ) = 0;
            % param_set.test_mask( test_n_start3_ind:test_n_end3_ind , 3:3:end ) = 0;

            % set which variational methods to run
            param_set.do_finite_lg_variational    = true;
            param_set.do_infinite_lg_variational  = false;

            % ---- advanced heuristics (may be of limited use) ---- %
            % search moves that combine apirs of features (eg sum and difference)
            param_set.try_search_heuristics   = true;     % try search moves during optimisation
            param_set.search_count            = 5;        % how many search moves per iteration
            param_set.search_heuristic        = 'random'; % 'random' pair,'full' - all pairs, 'correlation' - most correlated pair

            % adding data slowly (a form of tempering)
            param_set.add_data_in_blocks      = true;
            param_set.data_block_size         = floor(N/20);       % how many data points to add per iteration

            % reorder the features by popularity between optimisation iterations -
            % this improves the lower bound in the infinite approach.
            param_set.sort_feature_order      = false;     % recommended for infinite approaches

            % varying order of optimisation updates of the parameters.
            param_set.vary_update_order       = false;    % should order of parameter updates be varied in each optimisation iteration
            tic;
            % --------------------------%
            %       Run Inference       %
            % --------------------------%
            % finite LG variational
            if param_set.do_finite_lg_variational
                fprintf('#################################################\n');
                fprintf('Running Finite LG Variational\n');
                fprintf('#################################################\n');
                path(path, './vibp_finite/');
                param_set.use_finite_model = true;
                param_set.model_type = 'LG';
                t = cputime;
                model_finite_LG_variational = run_vibp( X , param_set );
                model_finite_LG_variational.time_elapsed = cputime - t;
                rmpath('./vibp_finite')
            end
            e_time=toc;
            % infinite LG variational
            % if param_set.do_infinite_lg_variational
            %     fprintf('#################################################\n');
            %     fprintf('Running Infinite LG Variational\n');
            %     fprintf('#################################################\n');
            %     path(path, './vibp_infinite/');
            %     param_set.use_finite_model = false;
            %     param_set.model_type = 'LG';
            %     t = cputime;
            %     model_infinite_LG_variational = run_vibp( X , param_set );
            %     model_infinite_LG_variational.time_elapsed = cputime - t;
            %     rmpath('./vibp_infinite')
            % end
            W_model=model_finite_LG_variational.best_model.phi_mean;
            W_model=W_model';
            size(W_model)
            Z_model=model_finite_LG_variational.best_model.nu;
            size(Z_model)
            % Z_model=Z_model(:,any(Z_model));
            size(Z_model)
            [~,getK]=size(Z_model);
            % W_model(getK+1:end,:)=[];
            size(W_model)
            [~,h_err] = best_fit(Z0,Z_model);
            rmse_err  =  norm(Z0*W0-Z_model*W_model,'fro')/(sqrt(sn*sd));
            rmse_n_err = norm(R-Z_model*W_model,'fro')/(sqrt(sn*sd)); 
            fprintf(file1,'%d %g\n',getK,h_err);
            fprintf(file2,'%d %g\n',getK,rmse_err);
            fprintf(file3,'%d %g\n',getK,rmse_n_err);       
            fprintf('set_K=%d,getK=%d,err=%f,training_time=%f\n',param_set.K,getK,rmse_n_err,e_time); 
        end      
    end     
end