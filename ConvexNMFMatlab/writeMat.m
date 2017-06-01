function  writeMat( mat, fpath )

[N,D] = size(mat);

fp = fopen(fpath,'w');
fprintf(fp, '%d %d\n', [N,D]);
for i = 1:N
	fprintf(fp, '%g ', mat(i,:));
	fprintf(fp, '\n');
end
fclose(fp);
