%% Solving the boolean Integer QP problem:
%%
%%     max  x'Qx 
%%     s.t. x_i \in {0,1}, i=1...n
%%
%% by reduction to the following IQP:
%% 
%%     max  y'Cy
%%     s.t. y_i^2 = 1, i=1...n

function [x,obj] = boolIQP(Q)

n = length(Q);

b = sum(Q,2)/4;

C = zeros(n+1,n+1);
C(1,1) = 0;
C(2:end,2:end) = Q/4;
C(1,2:end) = b';
C(2:end,1) = b;

[y, obj2] = diagEqIQP(C);
%['MaxCutObj=' num2str(obj2+sum(sum(Q))/4)]

if y(1) < 0
	y = -y;
end

x = (y(2:end)+1)/2;
obj = x'*Q*x;
['MaxCutObj=' num2str(obj)]
