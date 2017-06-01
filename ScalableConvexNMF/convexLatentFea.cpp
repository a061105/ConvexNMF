#include "ColumnGen/IQPSolve.h"
#include "ColumnGen/LowRankFunc.h"
#include "DualSub/dualsubsolve_dense.h"
#include "DualSub/Parser.h"
#include <Eigen/Dense>

using namespace std;

double primalObj( Matrix& Rt, Matrix& Zc, Matrix& V, Vector& c, int N, int D, int K, double lambda ){
	
	double obj = 0.0;
	for(int j=0;j<D;j++){
		for(int i=0;i<N;i++){
			double pred = 0.0;
			for(int k=0;k<K;k++)
				pred += Zc[i][k]*V[j][k];

			double tmp = ( Rt[j][i] - pred );
			obj += tmp*tmp;
		}
	}
	obj /= 2.0;
	
	for(int j=0;j<D;j++)
		for(int k=0;k<K;k++)
			obj += V[j][k]*V[j][k]/2.0;

	double sum = 0.0;
	for(int k=0;k<K;k++)
		sum += c[k];
	
	obj += lambda*sum;

	return obj;
}

double prox( double ck, double lambda ){
	
	if( ck < lambda )
		return 0.0;
	else
		return ck - lambda;
}

void convexLatentFea(Matrix& Rt, int N, int D, double lambda, Vector& c, Matrix& Z, Matrix& V){
	
	double TOL = 1e-2;
	Z.clear();
	Z.resize(N);
	c.clear();
	V.resize(D);
	
	Matrix A;
	A.resize(N);
	for(int i=0;i<N;i++)
		A[i].resize(D);
	
	int max_iter = 100;
	int max_inner_iter = 100;
	int max_A_iter = 100;
	int iter = 0;
	while( iter < max_iter ){
		
		cerr << "iter=" << iter << endl;
		
		Matrix Zc;
		if( c.size() ==  0 ){
			Zc.resize(N);
			for(int i=0;i<N;i++)
				Zc[i].push_back(0.0);
			for(int j=0;j<D;j++)
				V[j].resize(1);
		}else{
			Zc = Z;
			for(int i=0;i<N;i++)
				for(int k=0;k<c.size();k++)
					Zc[i][k]*=sqrt(c[k]);
			for(int j=0;j<D;j++)
				V[j].resize(c.size());
		}
		
		cerr << "coordinate solve..." << endl;
		coordinate_solver( Rt, Zc, max_A_iter, 1.0, V, A, N, D, c.size() );

		//print
		cerr << "primal obj=" << primalObj( Rt, Zc, V, c, N, D, c.size(), lambda ) 
			<< ", K=" << c.size() 
			<< endl;
		
		for(int k=0;k<c.size();k++)
			cerr << c[k] << " ";
		cerr << endl;

		//find greedy direction
		int SDP_K = 30;
		ExtensibleFunction* fun = new LowRankFunc(N, D, SDP_K, A);
		IQPSolve* solve = new IQPSolve();
		Vector z;
		solve->solve(fun, z);
		delete fun;
		
		for(int i=0;i<N;i++)
			cerr << z[i] << " ";
		cerr << endl;

		c.push_back(0.0);
		for(int i=0;i<N;i++)
			Z[i].push_back(z[i]);
		
		//tune weight
		Vector H_diag;
		H_diag.resize(c.size(),0.0);
		for(int i=0;i<N;i++){
			for(int k=0;k<c.size();k++){
				H_diag[k] += Z[i][k]*Z[i][k];
			}
		}
		for(int k=0;k<c.size();k++)
			H_diag[k] = H_diag[k]*H_diag[k];
		double max_Hii = -1e300;
		for(int k=0;k<c.size();k++)
			if( H_diag[k] > max_Hii )
				max_Hii = H_diag[k];

		double eta = 1e-2/(max_Hii*c.size());
		int K = c.size();
		for(int t2=0;t2<max_inner_iter*K;t2++){
			
			//grad_c
			Vector grad_c;
			grad_c.resize(c.size());
			Matrix AtZ;
			AtZ.resize(D);
			for(int j=0;j<D;j++){
				AtZ[j].clear();
				AtZ[j].resize(c.size(),0.0);
			}
			for(int i=0;i<N;i++){
				for(int j=0;j<D;j++){
					for(int k=0;k<c.size();k++){
						AtZ[j][k] += A[i][j]*Z[i][k];
					}
				}
			}
			for(int k=0;k<c.size();k++)
				grad_c[k] = 0.0;
			
			for(int j=0;j<D;j++){
				for(int k=0;k<c.size();k++)
					grad_c[k] += AtZ[j][k]*AtZ[j][k];
			}
			for(int k=0;k<c.size();k++)
				grad_c[k] = -grad_c[k]/2.0;

			//for(int k=0;k<c.size();k++)
			//	cerr << "grad[" << k << "]=" << grad_c[k] << ", ";
			//cerr << endl;
			
			//prox
			for(int k=0;k<c.size();k++){
				c[k] = prox(c[k] - eta*grad_c[k], eta*lambda);
			}
			
			//find A*
			Zc = Z;
			for(int i=0;i<N;i++)
				for(int k=0;k<c.size();k++)
					Zc[i][k]*=sqrt(c[k]);
			for(int j=0;j<D;j++)
				V[j].resize(c.size());
				
			coordinate_solver( Rt, Zc, max_A_iter, 1.0, V, A, N, D, K );
		}
		
		Matrix Znew;
		Znew.resize(N);
		for(int i=0;i<N;i++){
			for(int k=0;k<c.size();k++){
				if( c[k] > TOL )
					Znew[i].push_back(Z[i][k]);
			}
		}
		Z = Znew;

		Vector cnew;
		for(int k=0;k<c.size();k++)
			if( c[k] > TOL )
				cnew.push_back(c[k]);
		c = cnew;
		
		iter++;
	}
}

vector<Instance*>* readMat(char* data_fpath, int& D){
	
	vector<Instance*>* data = new vector<Instance*>();
	
	ifstream fin(data_fpath);
	int N;
	fin >> N >> D;
	double val;
	for(int i=0;i<N;i++){
		Instance* ins = new Instance();
		ins->yi = 0;
		for(int j=0;j<D;j++){
			fin >> val;
			ins->xi.push_back(make_pair(j,val));
		}
		data->push_back(ins);
	}

	return data;
}

int main(int argc, char** argv){
	
	if( argc < 1+3 ){
		cerr << "convexlatentFea [data] [lambda] [R_mul]" << endl;
		exit(0);
	}
	
	char* data_fpath = argv[1];
	double lambda = atof(argv[2]);
	double R_mul = atof(argv[3]);
	
	int N, D;
	//vector<Instance*>* data = Parser::parseSVM(data_fpath,D);
	vector<Instance*>* data = readMat(data_fpath, D);
	N = data->size();
	
	Matrix Rt;
	Rt.resize(D);
	for(int j=0;j<D;j++){
		Rt[j].resize(N);
		for(int i=0;i<data->size();i++)
			Rt[j][i] = 0.0;
	}
	
	for(int i=0;i<data->size();i++){
		for(SparseVec::iterator it=data->at(i)->xi.begin(); it!=data->at(i)->xi.end(); it++){
			Rt[it->first][i] = it->second;
		}
	}
	//multiply R_mul
	for(int j=0;j<D;j++)
		for(int i=0;i<N;i++)
			Rt[j][i] *= R_mul;
	
	//solve
	cerr << "solve begin" << endl;
	Matrix Z, V;
	Vector c;
	convexLatentFea(Rt, N, D, lambda, c, Z, V);
	
	Matrix Zt;
	transpose( Z, N, c.size(), Zt);

	if( c.size() > 0 ){
		
		vector<int> index;
		for(int i=0;i<c.size();i++)
			index.push_back(i);

		sort(index.begin(), index.end(), ScoreComp(&c[0]));

		for(vector<int>::iterator it=index.begin(); it!=index.end(); it++){
			cerr << "c_k=" << c[*it] << ": ";
			for(int i=0;i<N;i++)
				cerr << Zt[*it][i] << " ";
			cerr << endl;
		}
	}
}
