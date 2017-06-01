function [W,Z,R] = simulateMixReg(N,D,K,noise)

%K by D
W = rand(K,D);
Z = zeros(N,K);
for i = 1:N
				k = ceil(rand*K);
				Z(i,k) = 1;
end

R = Z*W + noise*randn(N,D);
