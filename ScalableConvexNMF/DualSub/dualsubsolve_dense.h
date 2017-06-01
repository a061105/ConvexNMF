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
inline double* get_sparse_row(vector< Instance* >* data ,int row ,int position,int numFea){
   int N = data->size();
   if(row==1){
   		double* v = new double[numFea];
    	Instance* ins = data->at(position);
    	for(int i=0;i<numFea;i++)
    		v[i]=0.0;
    	for(int i=0;i<ins->xi.size();i++){			
			int index = ins->xi[i].first;
			double value = ins->xi[i].second;
			v[index-1]=value;			
		} 		 
		return v;  	
    }   
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
			int index = ins->xi[j].first-1;
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

double pure_dot_x_nng( vector<double>& v, double* x ,int numFea){
	double sum=0.0;
	for(int i=0;i<numFea;i++){
		double xi = *(x+i);
		if( xi > 0.0 )
			sum += v[i]*xi;
	}	
	return sum;
}


/** Rt: D*N
 *  Z: N*K
 *  V: D*K
 *  A: N*D
 */
void coordinate_solver(Matrix& Rt, Matrix& Z,int max_iter,double lambda, Matrix& V, Matrix& A, int N,int D, int K){
	
	//calculate Hessian diagonal
	double* H_bound = new double [N];
	for(int i=0;i<N;i++){
		double square_sum = 0.0;
		for(int j=0;j<K;j++){
			square_sum += Z[i][j]*Z[i][j];
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
		//reach R_u
		while(iter < max_iter){


			//start inner loop
			for(int r=0;r<N;r++){ 
				//choose random gradient to update
				int i = index[r];
				//1. compute gradient of i 
				double gi = (1.0/lambda)*pure_dot_x_nng(Z[i],w,K)-Rt[u][i]+alpha[i];
				//2. compute alpha_u_i
				double new_alpha = alpha[i]-gi/H_bound[i];
				//3. maintain w
				double alpha_diff = new_alpha-alpha[i];
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o] = w[o]+alpha_diff*Z[i][o];
					}			
					alpha[i] = new_alpha;
				}				
			}

			//if(iter%10==0)
			//computer object_value, overhead?
			//if(iter%10==0){
			/*double* temp = new double[K];				
			for(int i=0;i<K;i++){
				temp[i]=0.0;
			}			
			for(int i=0;i<N;i++){
				for(int j=0;j<K;j++){
					temp[j]=temp[j]+alpha[i]*(*(Z+i*K+j));					
				}
			}
			//take positive
			for(int j=0;j<K;j++){
				temp[j]=max(0.0,temp[j]);
			}
			object_value=(0.5/lambda)*pure_dot(temp,temp,K)-pure_dot(R+u*N,alpha,N)+0.5*pure_dot(alpha,alpha,N);

			delete[] temp;
			// }
			cout <<"D="<<u<<", iter=" << iter <<" ,object_value="<<object_value<<endl ;				      */
			shuffle(index);
			iter++;
		}

		
		//cerr << endl;
		//output u_th model
		//update V;
		for(int i=0;i<K;i++)
			V[u][i]= max(0.0, w[i]);
			//*(V+i+u*K) = i + u*K;
		for(int i=0;i<N;i++)
			A[i][u] = alpha[i];
	}
	//release memory
	// for(int i=0;i<K;i++){
	// 	for(int j=0;j<D;j++){
	// 		printf("V[%d][%d]=%lf ",i,j,*(V+i*D+j));
	// 	}
	// 	cout<<endl;
	// }
	//*objCD = get_objvalue_d(R,V,Z,N,K,D);
	
	//delete[] Z;
	delete[] w;
	delete[] alpha;
	delete[] H_bound;
}
