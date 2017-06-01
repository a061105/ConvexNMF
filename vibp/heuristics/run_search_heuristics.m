function [model lower_bounds_log] = ...
    run_search_heuristics( model , X , lower_bounds_log , ...
    alpha , sigma_n , sigma_a , param_set )
% function [model lower_bounds_log] = ...
%     run_search_heuristics( model , X , lower_bounds_log , ...
%     alpha , sigma_n , sigma_a , param_set );
%
% This function tries out one of several different search move
% proposals, accepting them if they increase the lower bound,
% rejecting them otherwise.

% get K
K = param_set.K;

% loop for number of searches to try
for search_ind = 1:param_set.search_count
    switch param_set.search_heuristic

        % --- TRY ALL PAIRS --- %
        case 'full'
            maxLB = lower_bounds_log(1,end);
            found_better = false;

            % loop over all pairs
            for h=1:K
                for l=[1:(h-1) (h+1):K]

                    % initialise proposal
                    new_model = model_search_move(model, h, l, ceil( rand * 5 ) );

                    % compute update and store results
                    [new_model nlbl] = vibp(X, alpha, sigma_a, ...
                        sigma_n, new_model, ...
                        lower_bounds_log(:,end), param_set);

                    % if the proposal is better than past ones, store it
                    if nlbl(1,end) > maxLB + 0.1;
                        maxLB = nlbl(1,end);
                        best_model = new_model;
                        best_lbl = nlbl(:,2:end);
                        found_better = true;
                    end
                end % l loop
            end % h loop

            % update if we had an improvement
            if found_better
                fprintf('   accepted full proposal\n');
                model = best_model;
                lower_bounds_log = [lower_bounds_log best_lbl];
            end

        % --- RANDOMLY PROPOSE --- %
        case 'random'
            % compute proposal h and l
            h = ceil(rand() * K);
            l = ceil(rand() * K);
            while l == h
                l = ceil(rand() * K);
            end
            maxLB = lower_bounds_log(1,end);
            found_better = false;

            %  initialise proposal
            new_model = model_search_move(model, h, l, ceil( rand * 5 ) );

            % compute update and store results
            [new_model nlbl] = vibp(X, alpha, sigma_a, sigma_n, ...
                new_model, lower_bounds_log(:,end), param_set);

            % if the proposal is better than past ones, store it
            if nlbl(1,end) > maxLB
                maxLB = nlbl(1,end);
                best_model = new_model;
                best_lbl = nlbl(:,2:end);
                found_better = true;
            end

            % update if we had an improvement
            if found_better
                fprintf('   accepted random proposal\n');
                model = best_model;
                lower_bounds_log = [lower_bounds_log best_lbl];
            end

        % --- PROPOSE BASED ON CORRELATIONS --- %
        case 'correlation'

           % compute proposal assymetric correlation
            cc = zeros(K);
            for feat_i = 1:K
                for feat_j = 1:K
                    if feat_i == feat_j
                        cc( feat_i , feat_j ) = 0;
                    else
                        cc( feat_i , feat_j ) = model.nu( : , feat_i )' ...
                            * model.nu( : , feat_j ) ...
                            / norm( model.nu( : , feat_i ) );
                    end
                end
            end
            hl = sampleMultinomial( cc(:) );
            h = mod( hl , K );
            if h == 0
                h = K;
            end
            l = ceil(hl / K);

            % Now have the h,l sampled according to their correlation
            if h ~= l
                maxLB = lower_bounds_log(1,end);
                found_better = false;

                % initialise proposal
                new_model = model_search_move(model, h, l, ceil( rand * 5 ) );

                % compute update and store results
                [new_model nlbl] = vibp(X, alpha, sigma_a, sigma_n, ...
                    new_model, lower_bounds_log(:,end), param_set);

                % if the proposal is better than past ones, store it
                if nlbl(1,end) > maxLB
                    maxLB = nlbl(1,end);
                    best_model = new_model;
                    best_lbl = nlbl(:,2:end);
                    found_better = true;
                end

                % update if we had an improvement
                if found_better
                    fprintf('   accepted correlation proposal\n');
                    model = best_model;
                    lower_bounds_log = [lower_bounds_log best_lbl];
                end
            end

        otherwise
            fprintf('No search moves\n');

    end % end switch over heuristics
end % end search loop


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A helper function for the search moves
function new_model = model_search_move(original_model, h, l, search_move)
new_model = original_model;

if search_move > 5
    search_move = mod(search_move, 5);
    tmp = h;
    h = l;
    l = tmp;
end

switch search_move
    case 1
        % Let h = h - l, leave l alone
        new_model.phi_mean(:,h) = original_model.phi_mean(:,h) - original_model.phi_mean(:,l);
        new_model.nu(:,h) = max(original_model.nu(:,h) - original_model.nu(:,l), 0.01);

    case 2
        % Let h = h + l, leave l alone
        new_model.phi_mean(:,h) = original_model.phi_mean(:,h) + original_model.phi_mean(:,l);
        new_model.nu(:,h) = min(original_model.nu(:,h) + original_model.nu(:,l), 0.99);
        phi_cov_avg = (original_model.phi_cov(:,h) + original_model.phi_cov(:,l))/2;
        new_model.phi_cov(:,h) = phi_cov_avg;
        new_model.phi_cov(:,l) = phi_cov_avg;

    case 3
        % Let h = h + l, delete l
        new_model.phi_mean(:,h) = original_model.phi_mean(:,h) + original_model.phi_mean(:,l);
        new_model.phi_mean(:,l) = zeros(size(original_model.phi_mean(:,l)));
        new_model.nu(:,h) = min(original_model.nu(:,h) + original_model.nu(:,l), 0.99);
        new_model.nu(:,l) = 0.01;
        phi_cov_avg = (original_model.phi_cov(:,h) + original_model.phi_cov(:,l))/2;
        new_model.phi_cov(:,h) = phi_cov_avg;
        new_model.phi_cov(:,l) = phi_cov_avg;

    case 4
        % randomly reset h and l
        new_model.phi_mean(:,h) = randn(size(new_model.phi_mean(:,h)))*0.01;
        new_model.phi_mean(:,l) = randn(size(new_model.phi_mean(:,h)))*0.01;
        new_model.nu(:,h) = rand(size(new_model.nu(:,h)));
        new_model.nu(:,l) = rand(size(new_model.nu(:,h)));
        phi_cov_avg = (original_model.phi_cov(:,h) + original_model.phi_cov(:,l))/2;
        new_model.phi_cov(:,h) = phi_cov_avg;
        new_model.phi_cov(:,l) = phi_cov_avg;

    case 5
        % h = h - l, l = h + l
        new_model.phi_mean(:,h) = original_model.phi_mean(:,h) - original_model.phi_mean(:,l);
        new_model.phi_mean(:,l) = original_model.phi_mean(:,h) + original_model.phi_mean(:,l);
        new_model.nu(:,h) = (original_model.nu(:,h) + original_model.nu(:,l))/2;
        new_model.nu(:,l) = (original_model.nu(:,h) + original_model.nu(:,l))/2;
        phi_cov_avg = (original_model.phi_cov(:,h) + original_model.phi_cov(:,l))/2;
        new_model.phi_cov(:,h) = phi_cov_avg;
        new_model.phi_cov(:,l) = phi_cov_avg;
end

return
