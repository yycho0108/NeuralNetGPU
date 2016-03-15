/*
 * Net.cu
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */
#include "Net.h"

void view(std::string suffix, double* v, int n){
	std::cout << suffix << ":";
	double* v_c = (double*) malloc(n*sizeof(double));
	cudaMemcpy(v_c,v,n*sizeof(double),cudaMemcpyDeviceToHost);
	std::cout << '[';
	for(int i=0;i<n;++i){
		std::cout << v_c[i] << ' ';
	}
	std::cout << ']' << std::endl;
	free(v_c);
}


Net::Net(std::vector<int> topology): topology(topology)
{

	for(int i=0;i<topology.size();++i){
		layers.push_back(new Layer(topology[i]));
	}

	for(int i=1;i<topology.size();++i){
		double* pWT;
		cudaMalloc(&pWT, topology[i-1]*topology[i]*sizeof(double));
		randArr_g(topology[i-1],topology[i],pWT);
		WT.push_back(pWT);

		double* pw;
		cudaMalloc(&pw, topology[i]*topology[i-1]*sizeof(double));
		W.push_back(pw);

		double* pdw;
		cudaMalloc(&pdw, topology[i]*topology[i-1]*sizeof(double));
		dW.push_back(pdw);

		double* pmem;
		cudaMalloc(&pmem, topology[i]*sizeof(double));
		mem.push_back(pmem);

		//W.push_back(randArr(topology[i],topology[i-1]));//flattened
		//WT.push_back(std::vector<double>(topology[i]*topology[i-1]));
	}
	updateW();

	//view("W",W[0],topology[0]*topology[1]);
	//view("WT",WT[0],topology[0]*topology[1]);

}

Net::~Net() {
	for(int i=0;i<topology.size()-1;++i){
		cudaFree(WT[i]);
		cudaFree(W[i]);
		cudaFree(dW[i]);
		cudaFree(mem[i]);
	}
	for(auto& l : layers){
		delete l;
	}
}

std::vector<double> Net::FF(std::vector<double> X){

	layers.front()->setO(&X.front()); //host-to-device

	for(int i=1;i<layers.size();++i){
		auto& w = WT[i-1]; //cuda-ptr
		auto& o = layers[i-1]->getO();

		//TRANSFER
		auto& I = layers[i]->getI();
		auto& O = layers[i]->getO();

		int n = topology[i-1];
		int m = topology[i]; //may be reverse
		dot_g(w,o,m,1,n,I);
		sigmoid_g(I,O,topology[i]);

	}
	std::vector<double> O(topology.back());

	cudaMemcpy(&O.front(),layers.back()->getO(),topology.back()*sizeof(double),cudaMemcpyDeviceToHost);

	return O;
}

void Net::BP(std::vector<double> y){

	double* Y;
	cudaMalloc(&Y,y.size()*sizeof(double));
	cudaMemcpy(Y,&y.front(),y.size()*sizeof(double),cudaMemcpyHostToDevice);

	sub_g(Y,layers.back()->getO(),topology.back(),true);

	layers.back()->setG(Y);

	//view("lb", layers.back()->getG(),topology.back());

	for(int i=topology.size()-2;i>=1;--i){
		double*& G = layers[i+1]->getG(); //topology[i+1]
		double* d = dot_g(W[i],G,topology[i],1,topology[i+1],mem[i-1]); //may be wrong order
		//view("W", W[i], topology[i]*topology[i+1]);

		//view("d", d, topology[i]);

		double*& I = layers[i]->getI();
		sigmoidPrime_g(I,I,topology[i]); //s * (1.0-s)

		layers[i]->setG(mult_g(d,I,topology[i],true));

		//view("g", layers[i]->getG(),topology[i]);
		//cudaFree(d);
	}

	for(int i=1;i<topology.size();++i){
		auto& G = layers[i]->getG();
		auto& O = layers[i-1]->getO();
		auto dw = dot_g(G,O,topology[i],topology[i-1],1,dW[i-1]);

		mult_g(dw,0.6,topology[i]*topology[i-1],true); //assign
		add_g(WT[i-1],dw,topology[i]*topology[i-1],true); //assign;
	}
	updateW(); //update weight-transpose

	cudaFree(Y);
}

void Net::updateW(){ //update weight-transpose
	for(int i=1;i<topology.size();++i){
		transpose_g(WT[i-1],W[i-1],topology[i-1],topology[i]);
	}
}
