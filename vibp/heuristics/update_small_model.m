function [ model X param_set orig_data ] = update_small_model( model , X , param_set , orig_data )

% store back into orig_data
orig_data.model.tau = model.tau;
orig_data.model.phi_mean = model.phi_mean;
orig_data.model.phi_cov = model.phi_cov;
orig_data.model.nu( find( orig_data.data_ind_set ) , : ) = model.nu;
if strcmp( param_set.model_type , 'iICA' )
    orig_data.model.eta( find( orig_data.data_ind_set ) , : ) = model.eta;
    orig_data.model.mu( find( orig_data.data_ind_set ) , : ) = model.mu;
end

% take out part of the model
data_count = size( X , 1 );
N = size( orig_data.X , 1 );
if (data_count < N)
    remove_count = floor( param_set.data_block_size / 2 );
    remove_ind = randsample( find( orig_data.data_ind_set == 1 ) , remove_count );
    orig_data.data_ind_set( remove_ind ) = 0;
    add_ind = randsample( find( orig_data.data_ind_set == 0 ), param_set.data_block_size + remove_count );
    orig_data.data_ind_set( add_ind ) = 1;
    X = orig_data.X( find( orig_data.data_ind_set ) , : );
    model.nu = orig_data.model.nu( find( orig_data.data_ind_set ) , : );
    param_set.test_mask = param_set.orig_test_mask( find( orig_data.data_ind_set ) , : );
end
