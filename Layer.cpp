/*
 * Layer.cpp
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */

#include "Layer.h"
#include <cuda.h>

extern void sigmoid(std::vector<double>& i, std::vector<double>& o);


Layer::Layer(int n):
	I(n),O(n),G(n){

}

Layer::~Layer() {
	// TODO Auto-generated destructor stub
}

std::vector<double>& Layer::transfer(std::vector<double> i){
	I.swap(i);
	sigmoid(I,O);
	return O;
}

void Layer::setI(std::vector<double> i){
	I.swap(i);
}
void Layer::setO(std::vector<double> o){
	O.swap(o);
}
void Layer::setG(std::vector<double> g){
	G.swap(g);
}

std::vector<double>& Layer::getI(){
	return I;
}
std::vector<double>& Layer::getO(){
	return O;
}
std::vector<double>& Layer::getG(){
	return G;
}
