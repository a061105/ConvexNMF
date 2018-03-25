% tr( (G+A)*(G+A)' M ) = \|Z'(G+A)\|_F^2

function loss = DualLoss(R, Z, A, Tau)

%tmp = max(Z'*A,0.0);
tmp = Z'*A;
loss = -sum(sum(tmp.*tmp))/2/Tau + sum(sum(R.*A)) - sum(sum(A.*A))/2;
