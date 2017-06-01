function c2 = prox( c, lambda )
	
	c2 = c;
	c2(c<=lambda)=0;
	c2(c>lambda) = c(c>lambda)-lambda;
end
