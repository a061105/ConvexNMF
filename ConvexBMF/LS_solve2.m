function [W,grad_c] = LS_solve2( ZTR, ZTZ, c, Tau )

[K,D] = size(ZTR);

W = (ZTZ + Tau*diag(1./c)) \ ZTR;

ZTA = (ZTR - ZTZ*W);

grad_c = zeros(K,1);
for i = 1:K
	grad_c(i) = ZTA(i,:)*ZTA(i,:)'/(-2)/Tau;
end
