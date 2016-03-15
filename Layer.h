/*
 * Layer.h
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utils.h"

class Layer {
private:
	int n;
	double *I, *O, *G;
	//std::vector<double> I; //input
	//std::vector<double> O; //output
	//std::vector<double> G; //gradient

public:
	Layer(int n);
	~Layer();
	double*& transfer(double*);

	void setI(double*);
	void setO(double*);
	void setG(double*);

	double*& getI();
	double*& getO();
	double*& getG();

	int size();
};

#endif /* LAYER_H_ */
