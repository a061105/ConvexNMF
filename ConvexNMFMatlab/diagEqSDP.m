%%
%% Solve the problem:
%%
%%     max  Tr C'X
%%     s.t. diag(X) = e
%%

function [X,objval] = diagEqSDP(C)

n = length(C);

%prepare C
C = -C; %sqlp solves min Tr C'X

%prepare b
b = ones(n,1);
blk{1,1} = 's';  blk{1,2} = n;

%prepare A
A = cell(1,n);
for k = 1:n; A{k} = sparse(k,k,1,n,n); end;
Avec = svec(blk,A,ones(size(blk,1),1));

%initialize
[X0,y0,Z0] = infeaspt(blk,Avec,C,b);

%solve
[obj,X,y,Z] = sqlp(blk,Avec,C,b,[],X0,y0,Z0);
objval = -obj(1);
X = X{1};
