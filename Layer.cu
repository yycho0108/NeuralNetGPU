/*
 * Layer.cu
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */

#include "Layer.h"

Layer::Layer(int n):
	n(n){
	cudaMalloc(&I, n*sizeof(double));
	cudaMalloc(&O, n*sizeof(double));
	cudaMalloc(&G, n*sizeof(double));
}

Layer::~Layer() {
	cudaFree(I);
	cudaFree(O);
	cudaFree(G);
}

double*& Layer::transfer(double* i){
	cudaMemcpy(I,i,n*sizeof(double),cudaMemcpyDeviceToDevice);
	sigmoid_g(I,O,n);
	return O;
}

void Layer::setI(double* i){
	cudaMemcpy(I,i,n*sizeof(double),cudaMemcpyDeviceToDevice);

}

void Layer::setO(double* o){
	cudaMemcpy(O,o,n*sizeof(double),cudaMemcpyHostToDevice);
}


void Layer::setG(double* g){
	cudaMemcpy(G,g,n*sizeof(double),cudaMemcpyDeviceToDevice);
}

double*& Layer::getI(){
	return I;
}

double*& Layer::getO(){
	return O;
}

double*& Layer::getG(){
	return G;
}

int Layer::size(){
	return n;
}
