function [A,Z] = BPmeans( X, K )

[N,D] = size(X);

T = 10;
Z = round(rand(N,K));
A = inv(Z'*Z)*Z'*X;

last_obj = 1e300;
for t=1:T
	
	for k = 1:K
		
		obj = objective(X,Z,A);
		if obj < last_obj
			['obj=' num2str(obj)]
			last_obj = obj;
			pause(0.2);
		end

		for i = 1:N
			Z_try = Z;
			Z_try(i,k) = 1-Z(i,k);
			A_try = inv(Z_try'*Z_try)*Z_try'*X;
			
			if( objective(X,Z_try,A_try) < objective(X,Z,A) )
				Z = Z_try;
				A = A_try;
			end
		end
	end
end

end

function obj = objective(X,Z,A)

	tmp = X-Z*A;
	obj = sum(sum(tmp.*tmp));
end
