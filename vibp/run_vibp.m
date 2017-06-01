function out = run_vibp( X , param_set )
% function out = run_vibp( X , param_set )
% Inputs and Outputs:
%  * X is an N by D matrix of data
%  * param_set is a struct containing parameters for running the algorithm
%  (see tester.m for definitions)
%  * out is a struct containing 
%    - out.model_set = model_set;
%    - out.lower_bounds_log_set = lower_bounds_log_set;
%    - out.final_lower_bound_set = final_lower_bound_set;
%    - out.best_model = model;
%    - out.best_lower_bounds_log = lower_bounds_log;
%
% Notes:
%  * the tau parameters are variational parameters on the beta distribution 
%    over either pi (finite model) or v (infinite model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get basic info
K = param_set.K;
[N D] = size( X );
alpha = param_set.alpha;

% loop for the number of restarts
model_set = cell(1,param_set.restart_count);
lower_bounds_log_set = cell(1,param_set.restart_count);
final_lower_bound_set = zeros(1,param_set.restart_count);
for restart_index = 1:param_set.restart_count
    fprintf('\n----------------------------------------------\n');
    fprintf('Restart number %d of %d (%.2f%%)\n', restart_index, param_set.restart_count, ...
        restart_index/param_set.restart_count*100.0);

    % ---------------------------------------------------------- %
    %       Randomly initialize the variational parameters       %
    % ---------------------------------------------------------- %
    % You can often get better performance by doing a principled
    % initialisation, but this depends more on the model and data.
    clear model;

    % feature probabilities
    % finite model: pi's(k)  ~ Beta(tau(1,k),tau(2,k));
    % infinite model: v's(k) ~ Beta(tau(1,k),tau(2,k));
    model.tau = ones( 2 , K );
    if param_set.use_finite_model
        model.tau( 1 , : ) = alpha/K;
        model.tau = model.tau + ...
            .5 * min( 1 , alpha/K ) * ( rand( size( model.tau ) ) - .5 );
    else
        model.tau( 1 , : ) = alpha;
        model.tau = model.tau + ...
            .5 * min( 1 , alpha ) * ( rand( size( model.tau ) ) - .5 );
    end

    % features
    % A(k,:) ~ Normal(phi_mean(:,k), diag(phi_cov(:,k)))
    model.phi_mean = randn(D,K)*0.01;
    model.phi_cov  = randn(D,K).^2*0.1;

    % feature assignments
    % Z(n,k) ~ Bernoulli(nu(n,k))
    model.nu = rand(N,K);

    % initialise remaining variables
    initial_time = cputime;
    lower_bounds_log = zeros(2,0);
    lower_bounds_log(1,1) = compute_variational_lower_bound( ...
        param_set, X, alpha, param_set.sigma_a, param_set.sigma_n, model);
    lower_bounds_log(2,1) = 0;

    % initialise the model subset if adding data in pieces
    if param_set.add_data_in_blocks
        [ model X param_set orig_data ] = initialise_small_model( model , X , param_set );
    end

    % ------------------------------------------------- %
    %          Run the Variational Inference            %
    % ------------------------------------------------- %
    iter = 1;
    use_tempering = param_set.use_tempering;
    while 1

      % ---- store model parameters ---- %        
        model.tau_set{iter} = model.tau;
        model.phi_mean_set{iter} = model.phi_mean;
        model.phi_cov_set{iter} = model.phi_cov;
        model.nu_set{iter} = model.nu;

        % ---- slowly temper sigma_a and sigma_n if we are tempering ---- %
        % Note that by playing with the rate of tempering, you can
        % affect the quality of your final solution.  The following
        % rates are ones that we found useful with our data, but
        % might not be suitable for all data sets.
        if use_tempering
            sigma_n = param_set.sigma_n * (1 + 10 * exp(-iter/5));
            sigma_a = param_set.sigma_a * (1 + 2.5 * exp(-iter/5));
        else
            sigma_n = param_set.sigma_n;
            sigma_a = param_set.sigma_a;
        end

        % ---- run the variational optimisation updates ---- %
        initLB = compute_variational_lower_bound(...
            param_set, X, alpha, param_set.sigma_a, param_set.sigma_n, model );
        [ model, lower_bounds_log, num_iters] = vibp(...
            X, alpha, sigma_a, sigma_n, model, lower_bounds_log, param_set);
        lower_bounds_log(1,end+1) = compute_variational_lower_bound( ...
            param_set, X , alpha, param_set.sigma_a, param_set.sigma_n, model);
        lower_bounds_log(2,end) = 0;
        iter = iter + 1;

        % ---- apply various heuristic search moves to improve the variational lower bound ---- %
        % This applies various search moves to the parameters.
        if param_set.try_search_heuristics
            [ model lower_bounds_log ] = run_search_heuristics( model , X , lower_bounds_log , ...
                alpha , sigma_n , sigma_a , param_set );
        end

        % This incrementally adds data, letting the model learn the
        % parameters on a subset of the data and slowly adding more
        % data as we progress.
        if param_set.add_data_in_blocks
            [ model X param_set orig_data ] = update_small_model( model , X , param_set , orig_data );
        end

        % This reorders the features to be in order of more likely
        % to least likely.  This improves the lower bound in the
        % infinite approach, but does not affect the finite approach.
        if param_set.sort_feature_order
            [tmp feature_order] = sort(sum(model.nu), 'descend');
            model = update_feature_order( model , feature_order , param_set ); 
            if param_set.add_data_in_blocks
                orig_data.model = update_feature_order( orig_data.model , feature_order , param_set ); 
            end
        end

        % --- check stopping criteria ---- %
        % note: if tempering and we appear to have converged, turn
        % it off for a final untemperred iteration.
        if ( ( (lower_bounds_log(1,end) - initLB)  < param_set.stopping_thresh * abs(lower_bounds_log(1,end)) ) && ...
                ( size( X , 1 ) == N ) && (abs(sigma_n - param_set.sigma_n) < 0.01) &&...
                (abs(sigma_a - param_set.sigma_a) < 0.01))
            if use_tempering
                use_tempering = false;
            else
                break;
            end
        end

        % ---- print visual feedback ---- %
        fprintf('Large Iteration %3d (N = %d of %d, %3d inner itr - time: %.2f) lower bound: %10.2f (%10.3f improvement)  ',...
            iter, size( X , 1 ) , N, num_iters, cputime-initial_time, lower_bounds_log(1,end),...
            lower_bounds_log(1,end) - initLB);
        if use_tempering
            fprintf('tempered sigma_a: %0.2f sigma_n: %0.2f\n', sigma_a, sigma_n);
        else
            fprintf('\n');
        end
        if (cputime - initial_time) > 10000
            fprintf('Finished %d iterations\n', iter);
            break
        end
    end % loop over single restart
    fprintf('\n');

    % ---- store the results in a cell array for later analysis ---- %
    % If you do not care about intermediate values, this storage
    % increases the amount of memory needed and can slow down the
    % algorithm.  You might instead want to only pass back the final
    % values.
    model_set{ restart_index } = model;
    lower_bounds_log_set{ restart_index } = lower_bounds_log;
    final_lower_bound_set( restart_index ) = lower_bounds_log(1, end);

end % loop over restarts

% ---- find the best run ---- %
[ max_lb max_iter ] = max( final_lower_bound_set );
model = model_set{ max_iter };
lower_bounds_log = lower_bounds_log_set{ max_iter };
fprintf('\n----------------------------------------------\n');
fprintf('Best restart on training data: (%d, %.2f)\n\n', max_iter, max_lb);

% ---- store which run was best for the output ---- %
out.model_set = model_set;
out.lower_bounds_log_set = lower_bounds_log_set;
out.final_lower_bound_set = final_lower_bound_set;
out.best_model = model;
out.best_lower_bounds_log = lower_bounds_log;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display the predicted pi and Z, the features, and the predicted
% denoised data.
if param_set.show_final_plots

    % Get predicted Z
    Z_predicted = model.nu;
    if param_set.use_finite_model
        pi_predicted = model.tau(1,:) ./ sum( model.tau,1);
    else
        pi_predicted = v2pi( model.tau(1,:) ./ sum( model.tau,1) );
    end
    A_predicted = model.phi_mean;
    K_pred = size(Z_predicted, 2);

    % compute color scale
    image_scale = [ min( [ min(A_predicted(:)) min(X(:)) 0 ] ) ...
        max( [ max(A_predicted(:)) max(X(:)) 1 ] ) ];

    % Show the predicted features and their weights.  Sort in LOF
    % for easier visual comparison.
    figure(1)
    clf;
    [tmp sorted_indices] = sortrows((1-Z_predicted)');
    imagesc([pi_predicted(sorted_indices); Z_predicted(:,sorted_indices)]);
    colormap gray
    if K_pred > 0
        axis image;
    end
    title('Predicted \pi and Z');

    % Show the image features
    figure(2)
    clf;
    colormap gray;
    for k = 1:size(A_predicted,2)
        subplot(ceil(sqrt(size(A_predicted,2))),ceil(sqrt(size(A_predicted,2))),k);
        imagesc(reshape(A_predicted(:,sorted_indices(k)),4,4), image_scale );
    end
    title('Predicted image features');

    % Show the predicted underlying data uncorrupted by noise.
    figure(3)
    clf;
    colormap gray
    for n = 1:min([9 size(X,1)])
        subplot(3,3,n);
        if (K_pred > 0)
            imagesc(reshape(Z_predicted(n,:)*A_predicted',4,4), image_scale );
        end
    end
    title('Predicted denoised data');

    % Show the true data
    figure(4)
    clf;
    colormap gray
    for n = 1:min([9 size(X,1)])
        subplot(3,3,n);
        imagesc(reshape(X(n,:),4,4), image_scale );
    end
    title('True observed data');

    % Plot the LB
    figure(5);
    clf;
    hold on;
    iter_count = size(lower_bounds_log,2);
    legends = {'Lower bound'};
    plot( lower_bounds_log(1,:), 'k-');
    colors = {'c.', 'm.', 'g.', 'b.', 'r.'};
    labels = {'\tau optimizations', '\phi optimizations', '\nu optimizations', ...
        '\eta-\mu optimisations', 'iteration end', };
    for i = 1:5
        indices = find(lower_bounds_log(2,:) == i);
        if size(indices,2) > 0
            plot(indices, lower_bounds_log(1,indices), colors{i});
            legends{end+1} = labels{i};
        end
    end
    title('Lower bound on marginal likelihood');
    AX = legend(legends, 'Location', 'SouthEast');
    LEG = findobj(AX,'type','text');
    set(LEG,'FontSize',18);
    legend boxoff;
    fprintf(['Note that the lower bound in the plot might not' ...
             ' monotonically increase. This is due to tempering' ...
             ' and incremental introduction of the data.\n']);
    drawnow;
end

return

