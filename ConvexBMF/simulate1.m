function [W0,Z0,R] = simulate1(N,D,K, NOISE)

W0=randn(K,D); 
Z0=binornd(1,0.5,N,K);
R = Z0*W0 + NOISE*randn(N,D);
