clear
addpath('vibp_base/')
addpath('util/')
addpath('heuristics/')

% load data
load('data.mat')

% -------------------------- %
%       Set Parameters       %
% -------------------------- %
% set basic parameters of the data
param_set.alpha = alpha;                      % IBP concentration parameter
param_set.sigma_n = sigma_n;                  % noise variance
param_set.sigma_a = sigma_a;                  % feature variance
% set basic parameters of the algorithm
param_set.K                       = 6;        % truncation level
param_set.restart_count           = 5;        % number of random restarts
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
param_set.data_block_size         = 10;       % how many data points to add per iteration

% reorder the features by popularity between optimisation iterations -
% this improves the lower bound in the infinite approach.
param_set.sort_feature_order      = true;     % recommended for infinite approaches

% varying order of optimisation updates of the parameters.
param_set.vary_update_order       = false;    % should order of parameter updates be varied in each optimisation iteration

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

% infinite LG variational
if param_set.do_infinite_lg_variational
    fprintf('#################################################\n');
    fprintf('Running Infinite LG Variational\n');
    fprintf('#################################################\n');
    path(path, './vibp_infinite/');
    param_set.use_finite_model = false;
    param_set.model_type = 'LG';
    t = cputime;
    model_infinite_LG_variational = run_vibp( X , param_set );
    model_infinite_LG_variational.time_elapsed = cputime - t;
    rmpath('./vibp_infinite')
end


