function [sq_err,Z] = bestZErr(X, W)

[N,D] = size(X);
[K,D] = size(W);

%%%2^K by K matrix of enumaration
Zall = all01comb(K);
[Kall, K] = size(Zall);

Z = zeros(N,K);
sq_err = zeros(N,1);
for i = 1:N
		
				tmp=ones(Kall,1)*X(i,:)-Zall*W;
				err = sum( tmp .* tmp, 2 );
			 	[min_err,ind] = min(err);
				sq_err(i) = min_err;
				Z(i,:) = Zall(ind,:);
end
