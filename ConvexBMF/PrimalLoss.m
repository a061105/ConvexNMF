function loss = PrimalLoss(R, Z, V, Tau)

tmp= (R-Z*V);

loss = sum(sum( tmp.*tmp ))/2 + Tau*sum(sum(V.*V))/2;
