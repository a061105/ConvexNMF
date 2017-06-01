function [V,A] = LS_solve( R, Zc, Tau )

[N,K] = size(Zc);

V = (Zc'*Zc + Tau*eye(K)) \ (Zc'*R);
A = R-Zc*V;
