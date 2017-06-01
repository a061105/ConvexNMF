function ham_err = hammingErr(Z, Z0)

[N,K0] = size(Z0);
[N,K] = size(Z);

total_err = 0;
for k = 1:K0
		z = Z0(:,k);
		num_match = sum(Z == z*ones(1,K));
		err = 1 - (max(num_match) / N);
		total_err = total_err + err;
end
ham_err = total_err / K0;
