/*
 * Net.cpp
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */
#include "Net.h"

int sum(std::vector<int>& v){
	int s =0;
	for(auto& i : v){
		s += i;
	}
	return s;
}

Net::Net(std::vector<int> topology): topology(topology)
{

	for(int i=0;i<topology.size();++i){
		layers.push_back(Layer(topology[i]));
	}

	for(int i=1;i<topology.size();++i){
		wT.push_back(randArr(topology[i],topology[i-1]));
		weights.push_back(std::vector<double>(topology[i]*topology[i-1]));
		//weights.push_back(randArr(topology[i],topology[i-1]));//flattened
		//wT.push_back(std::vector<double>(topology[i]*topology[i-1]));
	}

	updatewT();
}

Net::~Net() {

}

std::vector<double>& Net::FF(std::vector<double> X){
	layers.front().setO(X);
	for(int i=1;i<layers.size();++i){
		auto& w = wT[i-1];
		auto& o = layers[i-1].getO();
		layers[i].transfer( dot(w,o,o.size()));
	}
	return layers.back().getO();
}


void Net::BP(std::vector<double> X, std::vector<double> Y){

	layers.back().setG(sub(Y,FF(X)));

	for(int i=topology.size()-2;i>=1;--i){
		auto& G = layers[i+1].getG();
		auto d = dot(weights[i],G,G.size());
		auto s = sigmoidPrime(layers[i].getI());
		layers[i].setG(mult(d,s));
	}

	for(int i=1;i<topology.size();++i){
		auto& G = layers[i].getG();
		auto& O = layers[i-1].getO();
		auto dW = dot(G,O,1);
		dW = mult(dW,0.6);
		wT[i-1] = add(wT[i-1],dW);
	}
	updatewT(); //update weight-transpose
}

void Net::updatewT(){ //update weight-transpose
	for(int i=1;i<topology.size();++i){
		transpose(wT[i-1],weights[i-1],topology[i-1],topology[i]);
	}
}
