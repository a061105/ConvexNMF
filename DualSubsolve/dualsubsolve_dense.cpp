#include "Parser.h"
#include "Instance.h"
#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include <omp.h>
#include <algorithm>

using namespace std;

template<class T>
void shuffle( vector<T>& vect ){
	
	int r;
	for(int i=0;i<vect.size()-1;i++){
		r =  (rand() % (vect.size() - i-1))+1+i ;
		swap(vect[i],vect[r]);
	}
}

double dot( double* v, vector<pair<int,double> >& x ){
	
	double sum=0.0;
	for(int i=0;i<x.size();i++){
	
		int index = x[i].first;
		double value = x[i].second;

		sum += v[index]*value;
	}	
	return sum;
}
double pure_dot( double* v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		sum += (*(v+i))*(*(x+i));
	}	
	return sum;
}
double pure_dot_x_nng( double* v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		double xi = *(x+i);
		if( xi > 0.0 )
			sum += (*(v+i))*xi;
	}	
	return sum;
}
inline double* get_sparse_row(vector< Instance* >* data ,int position,int numFea){
   int N = data->size();
   double* v = new double[numFea];
    	Instance* ins = data->at(position);
    	for(int i=0;i<numFea;i++)
    		v[i]=0.0;
    	for(int i=0;i<ins->xi.size();i++){			
			int index = ins->xi[i].first;
			double value = ins->xi[i].second;
			v[index]=value;			
		} 		 
	return v;  	
   
}
inline double get_objvalue(double** R, double** V, vector< Instance* >* Z,int N,int K,int D){
	double result=0.0;
	double* temp_row= new double[D];
    for(int i=0;i<N;i++){
    	for(int j=0;j<D;j++){
    		temp_row[j]=0.0;
     	}
    	Instance* ins = Z->at(i);
    	for(int j=0;j<ins->xi.size();j++){			
			int index = ins->xi[j].first;
			double value = ins->xi[j].second;
			for(int k=0;k<D;k++){
				temp_row[k]+=value*V[index][k];
			}	
		} 
		for(int j=0;j<D;j++){
    		result=result+(R[j][i]-temp_row[j])*(R[j][i]-temp_row[j]);
    	}

    }
    delete[] temp_row;
    return 0.5*result;
}
inline double get_objvalue_d(double* R, double* V, double* Z,int N,int K,int D){
	double result=0.0;
	double* temp_row= new double[D];
    for(int i=0;i<N;i++){
    	for(int j=0;j<D;j++){
    		temp_row[j]=0.0;
     	}
    	for(int j=0;j<K;j++){			
			for(int k=0;k<D;k++){
				temp_row[k]+=*(Z+i*K+j)*(*(V+j*D+k));
			}	
		} 
		for(int j=0;j<D;j++){
    		result=result+(*(R+j*N+i)-temp_row[j])*(*(R+j*N+i)-temp_row[j]);
    	}

    }
    delete[] temp_row;
    return 0.5*result;
}
void coordinate_solver(double* R,double* Z,int max_iter,double lambda, double* V, double* A ,double& objCD,int N,int D, int K){

  //calculate Hessian diagonal
  	double* H_bound = new double [N];
	for(int i=0;i<N;i++){
		double square_sum = 0.0;
		for(int j=0;j<K;j++){
		    square_sum += (*(Z+i*K+j))*(*(Z+i*K+j));
		}
		H_bound[i]=square_sum/lambda+1.0;
	}
	//initialize 
	double* w = new double[K];
	double* alpha = new double[N];
	double object_value;
	vector<int> index;
	for(int i=0;i<N;i++)
		index.push_back(i);
	int iter;
	//iteration
	for(int u=0;u<D;u++){	
		shuffle(index);
		iter=0;
		//initialize 
		for(int i=0;i<K;i++)
			w[i] = 0.0;
		for(int i=0;i<N;i++)
			alpha[i] = 0.0;
		object_value=0.0;
		while(iter < max_iter){
			//start inner loop
			for(int r=0;r<N;r++){ 
				//choose random gradient to update
				int i = index[r];
				//1. compute gradient of i 
				double gi = (1.0/lambda)*pure_dot_x_nng(Z+K*i,w,K)-*(R+u*N+i)+alpha[i];
				//2. compute alpha_u_i
				double new_alpha = alpha[i]-gi/H_bound[i];
				//3. maintain w
				double alpha_diff = new_alpha-alpha[i];
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o] = w[o]+alpha_diff*(*(Z+K*i+o));
					}			
					alpha[i] = new_alpha;
				}				
			}
     		shuffle(index);
			iter++;
		}
		for(int i=0;i<K;i++)
			*(V+i+u*K)= max(0.0, w[i]);
		for(int i=0;i<N;i++)
			*(A+i+u*N) = alpha[i];
	}
	objCD = get_objvalue_d(R,V,Z,N,K,D);
	delete[] w;
	delete[] alpha;
}
int main(int argc, char** argv){

	
	if( argc < 5 ){
		cerr << "Usage: ./dualsubsolve_dense [Z] [R] [lambda] [max_iter] (modelFile)\n";
		exit(0);
	}
	
	char* ZFile = argv[1];
    char* RFile = argv[2];
    double lambda = atof(argv[3]);
    int  max_iter=atoi(argv[4]);
	char* modelFile;
	if( argc >= 6 )
		modelFile = argv[5];
	else{
		modelFile = "model";
	}
	
	int D;
	int N;
	int K;
	vector<Instance*>* data_Z =  Parser::parseSVM(ZFile,K);
	vector<Instance*>* data_R =  Parser::parseSVM(RFile,D);
	N = data_Z->size();
	cerr << "N=" << N << endl;
	cerr << "D=" << D << endl;
	cerr << "K=" << K << endl;	
	double* w = new double[K];
	double* A = new double[N*D];
	double* R = new double[N*D];
	double* Z = new double[N*K];
	double* V =new double[K*D]; 
	double objCD;	
	//initialize R and Z to be dense matrix
	for (int j=0;j<D;j++){
		for(int i=0;i<N;i++)
			*(R+j*N+i)=0.0;
    }
    for (int j=0;j<N;j++){
		Instance* ins = data_R->at(j);
		 for(int i=0;i<ins->xi.size();i++){			
				int index = ins->xi[i].first;
				double value = ins->xi[i].second;
				*(R+(index)*N+j)=value;					
		    }
    } 
    for (int j=0;j<K;j++){
		for(int i=0;i<N;i++)
			*(Z+j*N+i)=0.0;
    }
    for (int j=0;j<N;j++){
		Instance* ins = data_Z->at(j);
		 for(int i=0;i<ins->xi.size();i++){			
				int index = ins->xi[i].first;
				double value = ins->xi[i].second;
				*(Z+j*K+index)=value;					
		    }
    }     
    coordinate_solver(R,Z,max_iter,lambda,V,A,objCD,N,D,K);  
    cout<<"Final result:"<<objCD<<endl;
	
}
