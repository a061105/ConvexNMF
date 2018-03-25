function [W0,Z0,R] = simulate2(N, d1,d2,p1,p2, K, NOISE)

D = d1*d2;
W0 = zeros(K,D);
for k = 1:K
	%random pattern
	img = zeros(d1,d2);
	pos_r = ceil(rand*(d1-p1+1));
	pos_c = ceil(rand*(d2-p2+1));
	r_range = pos_r:pos_r+p1-1;
	c_range = pos_c:pos_c+p2-1;
	img( r_range, c_range ) = 0.5*randn( length(r_range), length(c_range) );

	%place into W
	W0(k,:) = reshape(img,[1,D]);

	imwrite(img+0.5, ['img_dir/W' num2str(k) '.png']);
end

Z0=binornd(1,0.5,N,K);
R = Z0*W0 + NOISE*randn(N,D);

for i = 1:15
	imwrite( reshape(R(i,:),[d1,d2]) +0.5 , ['img_dir/R' num2str(i) '.png']);
end
