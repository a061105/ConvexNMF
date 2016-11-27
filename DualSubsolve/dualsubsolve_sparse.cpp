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
		sum += v[i]*x[i];
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
vector< Instance* >* rdata(vector<Instance*>* data, int numFea){

        int n = data->size();
        vector< Instance* >* reverse_data = new vector< Instance* >();
        for(int j=0;j<numFea;j++){
                Instance* ins = new Instance();
                reverse_data->push_back( ins );
        }
        for(int i=0;i<n;i++){

                Instance* ins = data->at(i);
                for(int j=0;j<ins->xi.size();j++){
                        int index = ins->xi[j].first;
                        double value = ins->xi[j].second;
                        pair<int,double> pair;                        
                        pair.first = i+1;
                        pair.second = value;
                        reverse_data->at(index-1)->xi.push_back(pair);
                }
        }
        return reverse_data;
}
//index start at 1
int main(int argc, char** argv){

	if( argc < 4 ){
		cerr << "Usage: dualsubsolve [Z] [R] [lambda](modelFile)\n";
		exit(0);
	}
	
	char* ZFile = argv[1];
    char* RFile = argv[2];
    double lambda = atof(argv[3]);
	char* modelFile;
	if( argc >= 5 )
		modelFile = argv[4];
	else{
		modelFile = "model";
	}
	
	int D;
	int N;
	int K;
	vector<Instance*>* data_Z =  Parser::parseSVM(ZFile,K);
	vector<Instance*>* data_R =  Parser::parseSVM(RFile,D);
	data_R =rdata(data_R,D);
	N = data_Z->size();
	cerr << "N=" << N << endl;
	cerr << "D=" << D << endl;
	cerr << "K=" << K << endl;	
	double* w = new double[K];
	double* alpha = new double[N];
	//compute upper bound and initialization
	double* H_bound = new double [N];
	for(int i=0;i<N;i++){
		double square_sum = 0.0;
		Instance* ins = data_Z->at(i);
		for(int j=0;j<ins->xi.size();j++){
		    double value = ins->xi[j].second*ins->xi[j].second;
		    square_sum += value;
		}
		H_bound[i]=square_sum;
	}
	//Outer Loop D times
	ofstream fout(modelFile);
	for(int u=0;u<D;u++){

		vector<int> index;
		for(int i=0;i<N;i++)
			index.push_back(i);
		shuffle(index);
		int max_iter = 80;
		int iter=0;
		//initialize 
		for(int i=0;i<K;i++)
		  w[i] = 0.0;
		for(int i=0;i<N;i++)
		  alpha[i] = 0.0;
		double object_value=0.0;
		//reach R_u
		double *R= get_sparse_row(data_R,1,u,N); 
		while(iter < max_iter){	
		   	double update_time = -omp_get_wtime();
			//start inner loop
			for(int r=0;r<N;r++){ 
				//choose random gradient to update
				int i = index[r];
				double* Z=get_sparse_row(data_Z,1,i,K);
				//1. compute gradient of i 
				double gi = 1.0/lambda*pure_dot(Z,w,K)-R[i]+alpha[i];
				//2. compute alpha_u_i
				double new_alpha = alpha[i]-gi/H_bound[i];
				//3. maintain w
				double alpha_diff = new_alpha-alpha[i];
				if(  fabs(alpha_diff) > 1e-8 ){
					for(int o=0;o<K;o++){
						w[o]=max(0.0,w[o]+alpha_diff*Z[o]);
					}			
					alpha[i] = new_alpha;
				}				
				delete [] Z;
			}
			update_time += omp_get_wtime();
			//if(iter%10==0)
			//computer object_value, overhead?
			//if(iter%10==0){
				double* temp = new double[K];
				double* Z_temp;
				for(int i=0;i<K;i++){
					temp[i]=0.0;
				}			
				for(int i=0;i<N;i++){
					Z_temp=get_sparse_row(data_Z,1,i,K);
					for(int j=0;j<K;j++){
						temp[j]=temp[j]+alpha[i]*Z_temp[j];					
					}
				}
				//take positive
				for(int j=0;j<K;j++){
						temp[j]=max(0.0,temp[j]);
				}
				object_value=0.5/lambda*pure_dot(temp,temp,K)-pure_dot(R,alpha,N)+0.5*pure_dot(alpha,alpha,N);
				delete [] Z_temp;
			    delete [] temp;
		   // }
			cerr <<"D="<<u<<", iter=" << iter << ", time=" << update_time <<" ,object_value="<<object_value<<endl ;			
			shuffle(index);
			iter++;		
		}
		delete [] R;
		cerr << endl;
		//output model
	
	fout << u << endl;
	for(int i=0;i<N;i++)
		if( fabs(alpha[i]) > 1e-12 )
			fout << i << " " << alpha[i] << endl;
	
	} 	
	//release memory
	delete [] alpha;
	fout.close();
}