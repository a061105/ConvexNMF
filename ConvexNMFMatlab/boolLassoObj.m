function obj = boolLassoObj(R,lambda,M,c)
	
	D = R-M;
	obj = sum(sum(D.*D))/2 + lambda*sum(c);
end
