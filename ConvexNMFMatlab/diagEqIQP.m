%% Solving the problem:
%%
%%     max  x'Cx 
%%     s.t. x_i^2 = 1, i=1...n
%%
%% by solving SDP that maximizes tr(C'X) and then do max-cut-style randomized rounding.

function [max_x, max_obj] = diagEqIQP(C)

n = length(C);

[X, obj] = diagEqSDP(C);

V = chol(X);

max_obj = -1e300;
max_x = -1;
for t = 1:1000
	r = randn(n,1);
	r = r ./ norm(r);
	
	x = sign(V'*r);
	obj = x'*C*x;
	
	if obj > max_obj
		max_obj = obj;
		max_x = x;
	end
end
