function [ model X param_set orig_data ] = initialise_small_model( model , X , param_set )
% Initialises the observed data to be only a subset of the data and
% initialises the model with the corresponding parameters.

% store original things
param_set.orig_test_mask = param_set.test_mask;
orig_data.X = X;
orig_data.model = model;

% reduce observations in current model
data_ind_set = zeros( size( X , 1 ) , 1 );
data_ind_set(1:param_set.data_block_size) = 1;
X = X(1:param_set.data_block_size,:);
model.nu = model.nu(1:param_set.data_block_size,:);
param_set.test_mask = param_set.test_mask(1:param_set.data_block_size,:);

% store data set ind
orig_data.data_ind_set = data_ind_set;
